use crate::{
    interpreter::{ControlFlow, Interpreter, NativeFuncPtr, Value},
    token::Ident,
};

pub fn print<'cx>(
    _: &mut Interpreter<'cx>,
    values: Vec<Value<'cx>>,
) -> Result<Value<'cx>, ControlFlow<'cx>> {
    println!("{:?}", values);
    Ok(Value::Null)
}

pub static NATIVE_FUNCS: &[(Ident, NativeFuncPtr)] = &[(Ident("print"), print)];
