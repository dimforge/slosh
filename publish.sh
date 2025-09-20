#! /bin/bash

tmp=`mktemp -d`

echo $tmp

cp -r crates $tmp/.
cp Cargo.toml $tmp/.

cp -r src $tmp/crates/slosh2d/.
cp -r LICENSE $tmp/crates/slosh2d/.
cp -r README.md $tmp/crates/slosh2d/.
cp -r shaders $tmp/crates/slosh2d/.

cp -r src $tmp/crates/slosh3d/.
cp -r LICENSE $tmp/crates/slosh3d/.
cp -r README.md $tmp/crates/slosh3d/.
cp -r shaders $tmp/crates/slosh3d/.

# Publish slosh2d
cd $tmp/crates/slosh2d
ls
sed 's#\.\./\.\./src#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish

# Publish slosh3d
cd ../slosh3d
sed 's#\.\./\.\./src#src#g' ./Cargo.toml > ./Cargo.toml.new
mv Cargo.toml.new Cargo.toml
sed 's#\.\./\.\./shaders#shaders#g' ./src/lib.rs > ./src/lib.rs.new
mv src/lib.rs.new src/lib.rs
cargo publish

# Cleanup
rm -rf $tmp

