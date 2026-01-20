//! Parallel prefix sum (scan) implementation for GPU.

use nalgebra::DVector;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{BufferUsages, Shader, ShaderArgs};
use stensor::tensor::GpuTensor;

/// GPU compute kernels for parallel prefix sum.
///
/// This is a special variant that produces results as if a 0 was prepended
/// to the input vector. Used for computing particle index offsets in blocks.
#[derive(Shader)]
#[shader(module = "slosh::grid::prefix_sum")]
pub struct WgPrefixSum<B: Backend> {
    prefix_sum: GpuFunction<B>,
    add_data_grp: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct PrefixSumArgs<'a, B: Backend> {
    data: &'a GpuTensor<u32, B>,
    aux: &'a GpuTensor<u32, B>,
}

impl<B: Backend> WgPrefixSum<B> {
    // TODO: figure out a way to read this from the shader.
    const THREADS: u32 = 256;

    /// Computes parallel prefix sum on GPU data.
    ///
    /// Uses a multi-stage algorithm to handle arbitrary-length arrays.
    /// The result is equivalent to a CPU scan with a 0 prepended.
    ///
    /// # Arguments
    ///
    /// * `backend` - GPU backend
    /// * `pass` - Compute pass
    /// * `workspace` - Auxiliary buffers for multi-stage scan
    /// * `data` - Input/output buffer to scan (modified in-place)
    pub fn launch(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        workspace: &mut PrefixSumWorkspace<B>,
        data: &GpuTensor<u32, B>,
    ) -> Result<(), B::Error> {
        // If this assert fails, the kernel launches bellow must be changed because we are using
        // a fixed size for the shared memory currently.
        assert_eq!(
            Self::THREADS,
            256,
            "Internal error: prefix sum assumes a thread count equal to 256"
        );

        workspace.reserve(backend, data.len() as u32)?;

        let ngroups0 = workspace.stages[0].buffer.len() as u32;
        let aux0 = &workspace.stages[0].buffer;

        let args0 = PrefixSumArgs { data, aux: aux0 };
        self.prefix_sum
            .launch_grid(backend, pass, &args0, [ngroups0, 1, 1])?;

        for i in 0..workspace.num_stages - 1 {
            let ngroups = workspace.stages[i + 1].buffer.len() as u32;
            let buf = &workspace.stages[i].buffer;
            let aux = &workspace.stages[i + 1].buffer;

            let args = PrefixSumArgs { data: buf, aux };
            self.prefix_sum
                .launch_grid(backend, pass, &args, [ngroups, 1, 1])?;
        }

        if workspace.num_stages > 2 {
            for i in (0..workspace.num_stages - 2).rev() {
                let ngroups = workspace.stages[i + 1].buffer.len() as u32;
                let buf = &workspace.stages[i].buffer;
                let aux = &workspace.stages[i + 1].buffer;
                let args = PrefixSumArgs { data: buf, aux };
                self.add_data_grp
                    .launch_grid(backend, pass, &args, [ngroups, 1, 1])?;
            }
        }

        if workspace.num_stages > 1 {
            let args = PrefixSumArgs { data, aux: aux0 };
            self.add_data_grp
                .launch_grid(backend, pass, &args, [ngroups0, 1, 1])?;
        }

        Ok(())
    }

    /// CPU implementation of the prefix sum for testing/validation.
    ///
    /// Applies the same algorithm as the GPU version but on CPU.
    pub fn eval_cpu(&self, v: &mut DVector<u32>) {
        for i in 0..v.len() - 1 {
            v[i + 1] += v[i];
        }

        // NOTE: we actually have a special variant of the prefix-sum
        //       where the result is as if a 0 was appended to the input vector.
        for i in (1..v.len()).rev() {
            v[i] = v[i - 1];
        }

        v[0] = 0;
    }
}

struct PrefixSumStage<B: Backend> {
    capacity: u32,
    buffer: GpuTensor<u32, B>,
}

/// Workspace buffers for multi-stage prefix sum.
///
/// Stores auxiliary buffers needed for hierarchical scan of large arrays.
#[derive(Default)]
pub struct PrefixSumWorkspace<B: Backend> {
    stages: Vec<PrefixSumStage<B>>,
    num_stages: usize,
}

impl<B: Backend> PrefixSumWorkspace<B> {
    /// Creates an empty workspace.
    pub fn new() -> Self {
        Self {
            stages: vec![],
            num_stages: 0,
        }
    }

    /// Creates a workspace with capacity for the given buffer length.
    ///
    /// Allocates all necessary auxiliary buffers upfront.
    pub fn with_capacity(backend: &B, buffer_len: u32) -> Result<Self, B::Error> {
        let mut result = Self {
            stages: vec![],
            num_stages: 0,
        };
        result.reserve(backend, buffer_len)?;
        Ok(result)
    }

    /// Ensures workspace has capacity for the given buffer length.
    ///
    /// Reallocates auxiliary buffers if needed.
    pub fn reserve(&mut self, backend: &B, buffer_len: u32) -> Result<(), B::Error> {
        let mut stage_len = buffer_len.div_ceil(WgPrefixSum::<B>::THREADS);

        if self.stages.is_empty() || self.stages[0].capacity < stage_len {
            // Reinitialize the auxiliary buffers.
            self.stages.clear();

            while stage_len != 1 {
                let buffer = GpuTensor::vector(
                    backend,
                    DVector::<u32>::zeros(stage_len as usize),
                    BufferUsages::STORAGE,
                )?;
                self.stages.push(PrefixSumStage {
                    capacity: stage_len,
                    buffer,
                });

                stage_len = stage_len.div_ceil(WgPrefixSum::<B>::THREADS);
            }

            // The last stage always has only 1 element.
            self.stages.push(PrefixSumStage {
                capacity: 1,
                buffer: GpuTensor::vector(
                    backend,
                    DVector::<u32>::zeros(1),
                    BufferUsages::STORAGE,
                )?,
            });
            self.num_stages = self.stages.len();
        } else if self.stages[0].buffer.len() as u32 != stage_len {
            // The stages have big enough buffers, but we need to adjust their length.
            self.num_stages = 0;
            while stage_len != 1 {
                self.num_stages += 1;
                stage_len = stage_len.div_ceil(WgPrefixSum::<B>::THREADS);
            }

            // The last stage always has only 1 element.
            self.num_stages += 1;
        }

        Ok(())
    }

    /*
    pub fn read_max_scan_value(&mut self) -> cust::error::CudaResult<u32> {
        for stage in &self.stages {
            if stage.len == 1 {
                // This is the last stage, it contains the total sum.
                let mut value = [0u32];
                stage.buffer.index(0).copy_to(&mut value)?;
                return Ok(value[0]);
            }
        }

        panic!("The GPU prefix sum has not been initialized yet.")
    }
    */
}

/*
#[cfg(test)]
mod test {
    use super::{PrefixSumWorkspace, WgPrefixSum};
    use nalgebra::DVector;
    use slang_hal::gpu::GpuInstance;
    use slang_hal::kernel::CommandEncoderExt;
    use stensor::tensor::GpuVector;
    use slang_hal::Shader;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_prefix_sum() {
        const LEN: u32 = 15071;

        let gpu = GpuInstance::new().await.unwrap();
        let prefix_sum = WgPrefixSum::from_device(gpu.device()).unwrap();

        let inputs = vec![
            DVector::<u32>::from_fn(LEN as usize, |_, _| 1),
            DVector::<u32>::from_fn(LEN as usize, |i, _| i as u32),
            DVector::<u32>::new_random(LEN as usize).map(|e| e % 10_000),
        ];

        for v_cpu in inputs {
            let mut encoder = gpu.device().create_command_encoder(&Default::default());
            let v_gpu = GpuVector::init(
                gpu.device(),
                &v_cpu,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            );
            let staging = GpuVector::uninit(
                gpu.device(),
                v_cpu.len() as u32,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            );

            let mut workspace = PrefixSumWorkspace::with_capacity(gpu.device(), v_cpu.len() as u32);
            let mut pass = encoder.compute_pass("test", None);
            prefix_sum.dispatch(gpu.device(), &mut pass, &mut workspace, &v_gpu);
            drop(pass);
            queue.encode(&mut encoder, None);
            staging.copy_from(&mut encoder, &v_gpu);

            let t0 = std::time::Instant::now();
            gpu.queue().submit(Some(encoder.finish()));
            let gpu_result = staging.read(gpu.device()).await.unwrap();
            println!("Gpu time: {}", t0.elapsed().as_secs_f32());

            let mut cpu_result = v_cpu.clone();

            let t0 = std::time::Instant::now();
            prefix_sum.eval_cpu(&mut cpu_result);
            println!("Cpu time: {}", t0.elapsed().as_secs_f32());
            // println!("input: {:?}", v_cpu);
            // println!("cpu output: {:?}", cpu_result);
            // println!("gpu output: {:?}", gpu_result);

            assert_eq!(DVector::from(gpu_result), cpu_result);
        }
    }
}
*/
