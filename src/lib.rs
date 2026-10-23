//! Python Shell for Orbital Compiler
//!
//! The Python host of a served program: the project holds the served program (the app composed with
//! `std-http-server`), the language it runs against and a small entry point that hands both to the native
//! kernel extension (`orbital_kernel`), which serves the program.

#![allow(unexpected_cfgs)]

pub mod kernel_host;
mod shell;
pub mod shell_program;

pub use kernel_host::{KernelProject, PythonBackend};
pub use shell::PythonShell;
