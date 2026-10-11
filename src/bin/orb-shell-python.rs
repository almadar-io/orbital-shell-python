fn main() -> std::process::ExitCode {
    orbital_core::shell_protocol::run(&orbital_shell_python::shell_program::PythonProgram)
}
