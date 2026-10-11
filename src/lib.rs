//! Python Shell for Orbital Compiler
//!
//! This crate provides the Python/PyTorch shell implementation for the Orbital compiler.
//! It generates Python code for ML simulation with PyTorch backend, ideal for trait machine
//! simulation and neural network training.
//!
//! # Features
//!
//! - **Pydantic Entities**: Type-safe entity models with validation
//! - **State Machines**: Async trait handlers with state management
//! - **PyTorch Integration**: S-expression neural networks compiled to PyTorch
//! - **FastAPI Backend**: REST API for trait event handlers
//! - **Headless Simulation**: Batch training and evaluation support

#![allow(unexpected_cfgs)]

mod backend;
pub mod codegen;
mod generators;
mod shell;
pub mod shell_program;

pub use backend::PythonBackend;
pub use codegen::*;
pub use generators::*;
pub use shell::PythonShell;
