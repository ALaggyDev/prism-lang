use crate::bytecode::{NativeFunc, Value};

pub fn print(args: &[Value]) -> Value {
    println!("{:?}", args);
    Value::Null
}

pub static NATIVE_FUNCS: &[(&str, NativeFunc)] = &[("print", print)];
