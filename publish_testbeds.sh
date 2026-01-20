#! /bin/bash

tmp=`mktemp -d`

echo $tmp

cp -r crates $tmp/.
cp Cargo.toml $tmp/.

cp -r src_testbed $tmp/crates/slosh_testbed2d/.
cp -r LICENSE $tmp/crates/slosh_testbed2d/.
cp -r README.md $tmp/crates/slosh_testbed2d/.
cp -r shaders_testbed $tmp/crates/slosh_testbed2d/.

cp -r src_testbed $tmp/crates/slosh_testbed3d/.
cp -r LICENSE $tmp/crates/slosh_testbed3d/.
cp -r README.md $tmp/crates/slosh_testbed3d/.
cp -r shaders_testbed $tmp/crates/slosh_testbed3d/.

# Publish slosh_testbed2d
cd $tmp/crates/slosh_testbed2d
ls
sed 's#\.\./\.\./src_testbed#src_testbed#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders_testbed#shaders_testbed#g' ./src_testbed/lib.rs > ./src_testbed/lib.rs.new
mv src_testbed/lib.rs.new src_testbed/lib.rs
cargo publish --features runtime

# Publish slosh_testbed3d
cd ../slosh_testbed3d
sed 's#\.\./\.\./src_testbed#src_testbed#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders_testbed#shaders_testbed#g' ./src_testbed/lib.rs > ./src_testbed/lib.rs.new
mv src_testbed/lib.rs.new src_testbed/lib.rs
cargo publish --features runtime

# Cleanup
rm -rf $tmp

