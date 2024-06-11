use once_cell::sync::Lazy;

use crate::{
    interpreter::{ControlFlow, Interpreter, NativeFuncPtr, Value},
    token::Ident,
};

pub fn print(_: &mut Interpreter, values: Vec<Value>) -> Result<Value, ControlFlow> {
    println!("{:?}", values);
    Ok(Value::Null)
}

// TODO: Do not use Lazy and Vec!!!!!!!
pub static NATIVE_FUNCS: Lazy<Vec<(Ident, NativeFuncPtr)>> =
    Lazy::new(|| vec![(Ident("print".into()), print)]);
