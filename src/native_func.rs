use crate::interpreter::{ControlFlow, Interpreter, NativeFuncPtr, Value};

pub fn print(_: &mut Interpreter, values: Vec<Value>) -> Result<Value, ControlFlow> {
    println!("{:?}", values);
    Ok(Value::Null)
}

pub static NATIVE_FUNCS: &[(&str, NativeFuncPtr)] = &[("print", print)];
