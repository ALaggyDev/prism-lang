use std::{collections::HashMap, fmt};

use gc::{unsafe_empty_trace, Finalize, Gc, Trace};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u16)]
pub enum InstrKind {
    Copy,

    LoadConst,
    LoadGlobal,
    StoreGlobal,

    OpAdd,
    OpMinus,
    OpMultiply,
    OpDivide,

    OpAnd,
    OpOr,
    OpNot,

    CmpEqual,
    CmpNotEqual,
    CmpGreater,
    CmpLess,
    CmpGreaterOrEqual,
    CmpLessOrEqual,

    Jump,
    JumpIf,

    PackTuple,
    UnpackTuple,

    Call,
    Return,
}

#[derive(Copy, Clone, PartialEq, Eq, Finalize)]
pub struct Instr(u64);

unsafe impl Trace for Instr {
    unsafe_empty_trace!();
}

impl Instr {
    pub fn new_1(kind: InstrKind, a1: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16))
    }

    pub fn new_2(kind: InstrKind, a1: u16, a2: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16) | ((a2 as u64) << 32))
    }

    pub fn new_3(kind: InstrKind, a1: u16, a2: u16, a3: u16) -> Self {
        Self((kind as u64) | ((a1 as u64) << 16) | ((a2 as u64) << 32) | ((a3 as u64) << 48))
    }

    pub fn kind(self) -> InstrKind {
        unsafe { std::mem::transmute(self.0 as u16) }
    }

    pub fn a1(self) -> u16 {
        (self.0 >> 16) as u16
    }

    pub fn a2(self) -> u16 {
        (self.0 >> 32) as u16
    }

    pub fn a3(self) -> u16 {
        (self.0 >> 48) as u16
    }
}

impl fmt::Debug for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}({}, {}, {})",
            self.kind(),
            self.a1(),
            self.a2(),
            self.a3()
        )
    }
}

#[macro_export]
macro_rules! instr {
    ($kind: ident, $a1: expr) => {
        $crate::bytecode::Instr::new_1($crate::bytecode::InstrKind::$kind, $a1)
    };
    ($kind: ident, $a1: expr, $a2: expr) => {
        $crate::bytecode::Instr::new_2($crate::bytecode::InstrKind::$kind, $a1, $a2)
    };
    ($kind: ident, $a1: expr, $a2: expr, $a3: expr) => {
        $crate::bytecode::Instr::new_3($crate::bytecode::InstrKind::$kind, $a1, $a2, $a3)
    };
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
    pub ip: u16,
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

    pub fn fetch_instr(&self) -> Instr {
        self.code_obj.code[self.ip as usize]
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

        let instr = frame.fetch_instr();
        frame.ip += 1;

        match instr.kind() {
            InstrKind::Copy => {
                *frame.get_mut_slot(instr.a1()) = frame.get_slot(instr.a2()).clone();
            }

            InstrKind::LoadConst => {
                *frame.get_mut_slot(instr.a1()) =
                    frame.code_obj.consts[instr.a2() as usize].clone();
            }

            InstrKind::LoadGlobal => {
                let name = &frame.code_obj.global_names[instr.a2() as usize];

                *frame.get_mut_slot(instr.a1()) =
                    self.globals.get(name).expect("No global name!").clone();
            }

            InstrKind::StoreGlobal => {
                let name = &frame.code_obj.global_names[instr.a1() as usize];

                self.globals
                    .insert(name.clone(), frame.get_slot(instr.a2()).clone());
            }

            InstrKind::OpAdd => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_add(frame.get_slot(instr.a2()), frame.get_slot(instr.a3()));
            }

            InstrKind::OpMinus => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_minus(frame.get_slot(instr.a2()), frame.get_slot(instr.a3()));
            }

            InstrKind::OpMultiply => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_multiply(frame.get_slot(instr.a2()), frame.get_slot(instr.a3()));
            }

            InstrKind::OpDivide => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_divide(frame.get_slot(instr.a2()), frame.get_slot(instr.a3()));
            }

            InstrKind::OpAnd => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::op_and(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::OpOr => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::op_or(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::OpNot => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::Bool(Value::op_not(frame.get_slot(instr.a2())));
            }

            InstrKind::CmpEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_equal(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpNotEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_not_equal(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpGreater => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_greater(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpLess => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_less(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpGreaterOrEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_greater_or_equal(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpLessOrEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_less_or_equal(
                    frame.get_slot(instr.a2()),
                    frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::Jump => {
                frame.ip = instr.a1();
            }

            InstrKind::JumpIf => {
                if frame.get_slot(instr.a2()).is_truthy() {
                    frame.ip = instr.a1();
                }
            }

            InstrKind::PackTuple => {
                let tuple: Box<[Value]> =
                    (&frame.stack[instr.a2() as usize..(instr.a2() + instr.a3()) as usize]).into();
                *frame.get_mut_slot(instr.a1()) = Value::Tuple(Gc::new(tuple));
            }

            InstrKind::UnpackTuple => {
                let Value::Tuple(tuple) = frame.get_slot(instr.a2()) else {
                    panic!("Trying to unpack non-tuple");
                };
                let tuple = Gc::clone(tuple);

                if instr.a3() as usize != tuple.len() {
                    panic!("Tuple len not match.");
                }

                frame.stack[instr.a1() as usize..(instr.a1() + instr.a3()) as usize]
                    .clone_from_slice(&tuple);
            }

            InstrKind::Call => {
                let Value::Callable(callable) = frame.get_slot(instr.a1()) else {
                    panic!("Trying to call non-callable.");
                };

                let args =
                    &frame.stack[(instr.a2() + 1) as usize..(instr.a2() + 1 + instr.a3()) as usize];

                match callable {
                    // If we are calling a native function, we don't need to create a call frame
                    Callable::Native(native_func) => {
                        *frame.get_mut_slot(instr.a2()) = native_func(args);
                    }

                    Callable::Func(code_obj) => {
                        let new_frame = CallFrame::new(Gc::clone(code_obj), args);

                        frame.ret_slot = instr.a2();

                        self.frames.push(new_frame);
                    }
                }
            }

            InstrKind::Return => {
                let ret_val = frame.get_slot(instr.a1()).clone();

                self.frames.pop();

                if let Some(old_frame) = self.frames.last_mut() {
                    *old_frame.get_mut_slot(old_frame.ret_slot) = ret_val;
                } else {
                    self.result = Some(ret_val);
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
