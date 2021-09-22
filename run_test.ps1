cargo test
cargo build --release
Copy-Item ./target/release/quat_optimiz.dll ./quat_optimiz.pyd
py test.py
