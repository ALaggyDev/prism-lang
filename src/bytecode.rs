use std::collections::HashMap;

use gc::{Finalize, Gc, Trace};

#[derive(Clone, PartialEq, Eq, Debug, Trace, Finalize)]
pub enum Instr {
    Copy { dest: u16, src: u16 },

    LoadConst { dest: u16, index: u16 },
    LoadGlobal { dest: u16, index: u16 },
    StoreGlobal { src: u16, index: u16 },

    OpAdd { dest: u16, op1: u16, op2: u16 },
    OpMinus { dest: u16, op1: u16, op2: u16 },
    OpMultiply { dest: u16, op1: u16, op2: u16 },
    OpDivide { dest: u16, op1: u16, op2: u16 },

    OpAnd { dest: u16, op1: u16, op2: u16 },
    OpOr { dest: u16, op1: u16, op2: u16 },
    OpNot { dest: u16, op1: u16 },

    CmpEqual { dest: u16, op1: u16, op2: u16 },
    CmpNotEqual { dest: u16, op1: u16, op2: u16 },
    CmpGreater { dest: u16, op1: u16, op2: u16 },
    CmpLess { dest: u16, op1: u16, op2: u16 },
    CmpGreaterOrEqual { dest: u16, op1: u16, op2: u16 },
    CmpLessOrEqual { dest: u16, op1: u16, op2: u16 },

    Jump { dest: u32 },
    JumpIf { dest: u32, op: u16 },

    PackTuple { dest: u16, src: u16, len: u16 },
    UnpackTuple { src: u16, dest: u16, len: u16 },

    Call { func: u16, src: u16, arg_count: u16 },
    Return { src: u16 },
}

#[derive(Clone, Debug, PartialEq, Trace, Finalize)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(Gc<Box<str>>),
    Tuple(Gc<Box<[Value]>>),
    Callable(Callable),
}

#[derive(Clone, Debug, PartialEq, Trace, Finalize)]
pub struct CodeObject {
    pub code: Box<[Instr]>,
    pub consts: Box<[Value]>,
    pub global_names: Box<[Box<str>]>,
    pub name: Box<str>,
    pub stack_count: u16,
    pub arg_count: u16,
}

#[derive(Clone, PartialEq, Debug)]
pub struct CallFrame {
    pub code_obj: Gc<CodeObject>,
    pub stack: Box<[Value]>,
    pub ip: u32,
    pub ret_slot: u16,
}

pub type NativeFunc = fn(args: &[Value]) -> Value;

#[derive(Clone, Debug, PartialEq, Trace, Finalize)]
pub enum Callable {
    Native(#[unsafe_ignore_trace] NativeFunc),
    Func(Gc<CodeObject>),
}

#[derive(Clone, Debug)]
pub struct Vm {
    pub frames: Vec<CallFrame>,
    pub globals: HashMap<Box<str>, Value>,
    pub result: Option<Value>,
}

impl Value {
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(val) => *val,
            Value::Number(val) => *val != 0.0 && !(*val).is_nan(),
            Value::String(val) => !val.is_empty(),
            Value::Tuple(_) => true,
            Value::Callable(_) => true,
        }
    }

    pub fn op_add(&self, rhs: &Value) -> Value {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 + op2),
            (Value::String(op1), Value::String(op2)) => {
                Value::String(Gc::new(format!("{}{}", op1, op2).into()))
            }
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_minus(&self, rhs: &Value) -> Value {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 - op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_multiply(&self, rhs: &Value) -> Value {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 * op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_divide(&self, rhs: &Value) -> Value {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 / op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_and(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Bool(op1), Value::Bool(op2)) => *op1 && *op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_or(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Bool(op1), Value::Bool(op2)) => *op1 || *op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_not(&self) -> bool {
        match self {
            Value::Bool(op) => !op,
            _ => panic!("Unsupported operation."),
        }
    }

    pub fn cmp_equal(&self, rhs: &Self) -> bool {
        // Note: Make sure to update this table when a new value is added
        match (self, rhs) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(val_1), Value::Bool(val_2)) => val_1 == val_2,
            (Value::Number(val_1), Value::Number(val_2)) => val_1 == val_2,
            (Value::String(val_1), Value::String(val_2)) => val_1 == val_2,

            (Value::Tuple(tuple_1), Value::Tuple(tuple_2)) => {
                if tuple_1.len() == tuple_2.len() {
                    tuple_1
                        .iter()
                        .zip(tuple_2.iter())
                        .all(|(a, b)| Value::cmp_equal(a, b))
                } else {
                    false
                }
            }

            (
                Value::Callable(Callable::Native(func_1)),
                Value::Callable(Callable::Native(func_2)),
            ) => func_1 == func_2,

            (Value::Callable(Callable::Func(func_1)), Value::Callable(Callable::Func(func_2))) => {
                Gc::ptr_eq(func_1, func_2)
            }

            _ => false,
        }
    }

    pub fn cmp_not_equal(&self, rhs: &Self) -> bool {
        !Value::cmp_equal(self, rhs)
    }

    pub fn cmp_greater(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 > op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_less(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 < op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_greater_or_equal(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 >= op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_less_or_equal(&self, rhs: &Value) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 < op2,
            _ => panic!("Unsupported pair."),
        }
    }
}

impl CallFrame {
    pub fn new(code_obj: Gc<CodeObject>, args: &[Value]) -> Self {
        if code_obj.arg_count as usize != args.len() {
            panic!("Argument count mismatch!");
        }

        let mut stack = vec![Value::Null; code_obj.stack_count as usize].into_boxed_slice();
        stack[..args.len()].clone_from_slice(args);

        Self {
            code_obj,
            stack,
            ip: 0,
            ret_slot: 0,
        }
    }

    pub fn fetch_instr(&self) -> &Instr {
        &self.code_obj.code[self.ip as usize]
    }

    pub fn get_slot(&self, index: u16) -> &Value {
        &self.stack[index as usize]
    }

    pub fn get_mut_slot(&mut self, index: u16) -> &mut Value {
        &mut self.stack[index as usize]
    }
}

impl Vm {
    pub fn get_cur_frame(&self) -> &CallFrame {
        self.frames.last().unwrap()
    }

    pub fn get_mut_cur_frame(&mut self) -> &mut CallFrame {
        self.frames.last_mut().unwrap()
    }

    pub fn step(&mut self) {
        if self.result.is_some() {
            return;
        }

        let frame = self.frames.last_mut().unwrap();

        let instr = frame.fetch_instr().clone();
        frame.ip += 1;

        match instr {
            Instr::Copy { dest, src } => {
                *frame.get_mut_slot(dest) = frame.get_slot(src).clone();
            }

            Instr::LoadConst { dest, index } => {
                *frame.get_mut_slot(dest) = frame.code_obj.consts[index as usize].clone();
            }

            Instr::LoadGlobal { dest, index } => {
                let name = &frame.code_obj.global_names[index as usize];

                *frame.get_mut_slot(dest) =
                    self.globals.get(name).expect("No global name!").clone();
            }

            Instr::StoreGlobal { src, index } => {
                let name = &frame.code_obj.global_names[index as usize];

                self.globals
                    .insert(name.clone(), frame.get_slot(src).clone());
            }

            Instr::OpAdd { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) = Value::op_add(frame.get_slot(op1), frame.get_slot(op2));
            }

            Instr::OpMinus { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::op_minus(frame.get_slot(op1), frame.get_slot(op2));
            }

            Instr::OpMultiply { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::op_multiply(frame.get_slot(op1), frame.get_slot(op2));
            }

            Instr::OpDivide { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::op_divide(frame.get_slot(op1), frame.get_slot(op2));
            }

            Instr::OpAnd { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::Bool(Value::op_and(frame.get_slot(op1), frame.get_slot(op2)));
            }

            Instr::OpOr { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::Bool(Value::op_or(frame.get_slot(op1), frame.get_slot(op2)));
            }

            Instr::OpNot { dest, op1 } => {
                *frame.get_mut_slot(dest) = Value::Bool(Value::op_not(frame.get_slot(op1)));
            }

            Instr::CmpEqual { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::Bool(Value::cmp_equal(frame.get_slot(op1), frame.get_slot(op2)));
            }

            Instr::CmpNotEqual { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) = Value::Bool(Value::cmp_not_equal(
                    frame.get_slot(op1),
                    frame.get_slot(op2),
                ));
            }

            Instr::CmpGreater { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::Bool(Value::cmp_greater(frame.get_slot(op1), frame.get_slot(op2)));
            }

            Instr::CmpLess { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) =
                    Value::Bool(Value::cmp_less(frame.get_slot(op1), frame.get_slot(op2)));
            }

            Instr::CmpGreaterOrEqual { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) = Value::Bool(Value::cmp_greater_or_equal(
                    frame.get_slot(op1),
                    frame.get_slot(op2),
                ));
            }

            Instr::CmpLessOrEqual { dest, op1, op2 } => {
                *frame.get_mut_slot(dest) = Value::Bool(Value::cmp_less_or_equal(
                    frame.get_slot(op1),
                    frame.get_slot(op2),
                ));
            }

            Instr::Jump { dest } => {
                frame.ip = dest;
            }

            Instr::JumpIf { dest, op } => {
                if frame.get_slot(op).is_truthy() {
                    frame.ip = dest;
                }
            }

            Instr::PackTuple { dest, src, len } => {
                let tuple: Box<[Value]> = (&frame.stack[src as usize..(src + len) as usize]).into();
                *frame.get_mut_slot(dest) = Value::Tuple(Gc::new(tuple));
            }

            Instr::UnpackTuple { src, dest, len } => {
                let Value::Tuple(tuple) = frame.get_slot(src) else {
                    panic!("Trying to unpack non-tuple");
                };
                let tuple = Gc::clone(tuple);

                if len as usize != tuple.len() {
                    panic!("Tuple len not match.");
                }

                frame.stack[dest as usize..(dest + len) as usize].clone_from_slice(&tuple);
            }

            Instr::Call {
                func,
                src,
                arg_count,
            } => {
                let Value::Callable(callable) = frame.get_slot(func) else {
                    panic!("Trying to call non-callable.");
                };

                let args = &frame.stack[(src + 1) as usize..(src + 1 + arg_count) as usize];

                match callable {
                    // If we are calling a native function, we don't need to create a call frame
                    Callable::Native(native_func) => {
                        *frame.get_mut_slot(src) = native_func(args);
                    }

                    Callable::Func(code_obj) => {
                        let new_frame = CallFrame::new(Gc::clone(code_obj), args);

                        frame.ret_slot = src;

                        self.frames.push(new_frame);
                    }
                }
            }

            Instr::Return { src } => {
                let ret_val = frame.get_slot(src).clone();

                self.frames.pop();

                if let Some(old_frame) = self.frames.last_mut() {
                    *old_frame.get_mut_slot(old_frame.ret_slot) = ret_val;
                } else {
                    self.result = Some(ret_val);
                    return;
                }
            }
        }

        // // DEBUG ONLY
        // let frame = self.get_cur_frame();

        // println!(
        //     "{} {} {:?} {:?}",
        //     frame.code_obj.name,
        //     self.frames.len(),
        //     instr,
        //     frame.stack
        // );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use gc::Gc;

    use crate::native_func::NATIVE_FUNCS;

    use super::{CallFrame, Callable, CodeObject, Instr, Value, Vm};

    #[test]
    pub fn test_fibonacci() {
        let fib = Gc::new(CodeObject {
            #[rustfmt::skip]
            code: Box::new([
                    // load 1 & 2
                    Instr::LoadConst { dest: 1, index: 0 },
                    Instr::LoadConst { dest: 2, index: 1 },

                    // if n <= 1, return n
                    Instr::CmpGreater { dest: 3, op1: 0, op2: 1 },
                    Instr::JumpIf { dest: 5, op: 3 },

                    Instr::Return { src: 0 },

                    // load fib
                    Instr::LoadGlobal { dest: 3, index: 0 },

                    // call fib(n - 1)
                    Instr::OpMinus { dest: 5, op1: 0, op2: 1 },
                    Instr::Call { func: 3, src: 4, arg_count: 1 },

                    // call fib(n - 2)
                    Instr::OpMinus { dest: 7, op1: 0, op2: 2 },
                    Instr::Call { func: 3, src: 6, arg_count: 1 },

                    // return fib(n - 1) + fib(n - 2)
                    Instr::OpAdd { dest: 0, op1: 4, op2: 6 },
                    Instr::Return { src: 0 },
                ]),
            consts: Box::new([Value::Number(1.0), Value::Number(2.0)]),
            global_names: Box::new(["fib".into()]),
            name: "fib".into(),
            stack_count: 8,
            arg_count: 1,
        });

        let call_frame = CallFrame::new(Gc::clone(&fib), &[Value::Number(15.0)]);

        let mut vm = Vm {
            frames: vec![call_frame],
            globals: HashMap::new(),
            result: None,
        };

        vm.globals
            .insert("fib".into(), Value::Callable(Callable::Func(fib)));

        for (name, native_func) in NATIVE_FUNCS {
            vm.globals.insert(
                (*name).into(),
                Value::Callable(Callable::Native(*native_func)),
            );
        }

        while vm.result.is_none() {
            vm.step();
        }

        assert_eq!(vm.result, Some(Value::Number(610.0)));
    }
}
