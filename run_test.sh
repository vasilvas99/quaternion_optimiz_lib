#!/usr/bin/env bash
cargo test
cargo build --release
cp ./target/release/quat_optimiz.so ./quat_optimiz.so
python3 test.py
