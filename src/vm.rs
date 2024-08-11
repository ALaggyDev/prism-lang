use std::collections::HashMap;

use crate::{
    bytecode::{Instr, InstrKind},
    call_frame::CallFrames,
    native_func::NATIVE_FUNCS,
};

use gc_arena::{
    lock::{GcRefLock, RefLock},
    Arena, Collect, Gc, Mutation, Rootable,
};

#[derive(Copy, Clone, Debug, Collect)]
#[collect(no_drop)]
pub enum Value<'gc> {
    Null,
    Bool(bool),
    Number(f64),
    String(Gc<'gc, Box<str>>),
    Tuple(Gc<'gc, Box<[Value<'gc>]>>),
    Array(GcRefLock<'gc, Vec<Value<'gc>>>),
    // Dict(Gc<GcCell<HashMap<Gc<Box<str>>, Value<'gc>>>>),
    Callable(Callable<'gc>),
}

#[derive(Clone, Debug, Collect)]
#[collect(no_drop)]
pub struct CodeObject<'gc> {
    pub code: Box<[Instr]>,
    pub consts: Box<[Value<'gc>]>,
    pub global_names: Box<[Box<str>]>,
    pub stack_count: u16,
    pub arg_count: u16,
}

#[derive(Copy, Clone, Debug, Collect)]
#[collect(require_static)]
pub struct NativeFunc(pub for<'gc> fn(args: &[Value<'gc>], mc: &Mutation<'gc>) -> Value<'gc>);

#[derive(Copy, Clone, Debug, Collect)]
#[collect(no_drop)]
pub enum Callable<'gc> {
    Native(NativeFunc),
    Func(Gc<'gc, CodeObject<'gc>>),
}

#[derive(Clone, Debug, Collect)]
#[collect(no_drop)]
pub struct Vm<'gc> {
    pub frames: CallFrames<'gc>,
    pub globals: HashMap<Box<str>, Value<'gc>>,
    pub finished: bool,
}

fn float_to_u64(val: f64) -> Option<u64> {
    if val as u64 as f64 == val {
        Some(val as u64)
    } else {
        None
    }
}

impl<'gc> Value<'gc> {
    pub fn is_truthy(self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(val) => val,
            Value::Number(val) => val != 0.0 && !(val).is_nan(),
            Value::String(val) => !val.is_empty(),
            Value::Tuple(_) => true,
            Value::Array(arr) => !arr.borrow().is_empty(),
            // Value::Dict(_) => true,
            Value::Callable(_) => true,
        }
    }

    pub fn op_add(mc: &Mutation<'gc>, lhs: Value<'gc>, rhs: Value<'gc>) -> Value<'gc> {
        match (lhs, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 + op2),
            (Value::String(op1), Value::String(op2)) => {
                Value::String(Gc::new(mc, format!("{}{}", op1, op2).into()))
            }
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_minus(self, rhs: Value<'gc>) -> Value<'gc> {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 - op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_multiply(self, rhs: Value<'gc>) -> Value<'gc> {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 * op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_divide(self, rhs: Value<'gc>) -> Value<'gc> {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => Value::Number(op1 / op2),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_negate(self) -> Value<'gc> {
        match self {
            Value::Number(op) => Value::Number(-op),
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_and(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Bool(op1), Value::Bool(op2)) => op1 && op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_or(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Bool(op1), Value::Bool(op2)) => op1 || op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn op_not(self) -> bool {
        match self {
            Value::Bool(op) => !op,
            _ => panic!("Unsupported operation."),
        }
    }

    pub fn cmp_equal(self, rhs: Self) -> bool {
        // Note: Make sure to update this table when a new Value<'gc> is added
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
                        .all(|(a, b)| Value::cmp_equal(*a, *b))
                } else {
                    false
                }
            }

            (Value::Array(arr_1), Value::Array(arr_2)) => Gc::ptr_eq(arr_1, arr_2),

            // (Value::Dict(dict_1), Value::Dict(dict_2)) => Gc::ptr_eq(dict_1, dict_2),
            (
                Value::Callable(Callable::Native(NativeFunc(func_1))),
                Value::Callable(Callable::Native(NativeFunc(func_2))),
            ) => func_1 == func_2,

            (Value::Callable(Callable::Func(func_1)), Value::Callable(Callable::Func(func_2))) => {
                Gc::ptr_eq(func_1, func_2)
            }

            _ => false,
        }
    }

    pub fn cmp_not_equal(self, rhs: Self) -> bool {
        !Value::cmp_equal(self, rhs)
    }

    pub fn cmp_greater(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 > op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_less(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 < op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_greater_or_equal(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 >= op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn cmp_less_or_equal(self, rhs: Value<'gc>) -> bool {
        match (self, rhs) {
            (Value::Number(op1), Value::Number(op2)) => op1 <= op2,
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn load_index(mc: &Mutation<'gc>, orig: Value<'gc>, index: Value<'gc>) -> Value<'gc> {
        match (orig, index) {
            (Value::String(string), Value::Number(index)) => {
                let index = float_to_u64(index).expect("Not integer!") as usize;

                if let Some(chr) = string.chars().nth(index) {
                    let temp: String = chr.into();
                    Value::String(Gc::new(mc, temp.into()))
                } else {
                    panic!("Index out of range.");
                }
            }
            (Value::Tuple(tuple), Value::Number(index)) => {
                let index = float_to_u64(index).expect("Not integer!") as usize;

                tuple[index]
            }
            (Value::Array(array), Value::Number(index)) => {
                let index = float_to_u64(index).expect("Not integer!") as usize;

                array.borrow()[index]
            }
            // (Value::Dict(dict), Value::String(key)) => {
            //     dict.borrow().get(key).expect("Value<'gc> not found.").clone()
            // }
            _ => panic!("Unsupported pair."),
        }
    }

    pub fn store_index(mc: &Mutation<'gc>, orig: Value<'gc>, index: Value<'gc>, rhs: Value<'gc>) {
        match (orig, index) {
            (Value::Array(array), Value::Number(index)) => {
                let index = float_to_u64(index).expect("Not integer!") as usize;

                array.borrow_mut(mc)[index] = rhs;
            }
            // (Value::Dict(dict), Value::String(key)) => {
            //     dict.borrow().get(key).expect("Value<'gc> not found.").clone()
            // }
            _ => panic!("Unsupported pair."),
        }
    }
}

impl<'gc> Vm<'gc> {
    pub fn new(use_builtins: bool) -> Self {
        let mut vm = Vm {
            frames: CallFrames::new(),
            globals: HashMap::new(),
            finished: false,
        };

        if use_builtins {
            for (name, native_func) in NATIVE_FUNCS {
                vm.globals.insert(
                    (*name).into(),
                    Value::Callable(Callable::Native(*native_func)),
                );
            }
        }

        vm
    }

    pub fn step(&mut self, mc: &Mutation<'gc>) {
        if self.finished {
            return;
        }

        let frame = self.frames.get_last_mut();

        let Some(instr) = frame.fetch_instr() else {
            if self.frames.len() == 1 {
                // global context
                self.frames.pop_frame();
                self.finished = true;
                return;
            } else {
                panic!("Instruction pointer out of bound.");
            }
        };

        frame.ip += 1;

        match instr.kind() {
            InstrKind::Copy => {
                *frame.get_mut_slot(instr.a1()) = *frame.get_slot(instr.a2());
            }

            InstrKind::LoadConst => {
                *frame.get_mut_slot(instr.a1()) = frame.code_obj.consts[instr.a2() as usize];
            }

            InstrKind::LoadGlobal => {
                let name = &frame.code_obj.global_names[instr.a2() as usize];

                *frame.get_mut_slot(instr.a1()) = *self.globals.get(name).expect("No global name!");
            }

            InstrKind::StoreGlobal => {
                let name = &frame.code_obj.global_names[instr.a1() as usize];

                self.globals
                    .insert(name.clone(), *frame.get_slot(instr.a2()));
            }

            InstrKind::LoadIndex => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::load_index(mc, *frame.get_slot(instr.a2()), *frame.get_slot(instr.a3()));
            }

            InstrKind::StoreIndex => {
                Value::store_index(
                    mc,
                    *frame.get_slot(instr.a1()),
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                );
            }

            InstrKind::OpAdd => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_add(mc, *frame.get_slot(instr.a2()), *frame.get_slot(instr.a3()));
            }

            InstrKind::OpMinus => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_minus(*frame.get_slot(instr.a2()), *frame.get_slot(instr.a3()));
            }

            InstrKind::OpMultiply => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_multiply(*frame.get_slot(instr.a2()), *frame.get_slot(instr.a3()));
            }

            InstrKind::OpDivide => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::op_divide(*frame.get_slot(instr.a2()), *frame.get_slot(instr.a3()));
            }

            InstrKind::OpNegate => {
                *frame.get_mut_slot(instr.a1()) = Value::op_negate(*frame.get_slot(instr.a2()));
            }

            InstrKind::OpAnd => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::op_and(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::OpOr => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::op_or(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::OpNot => {
                *frame.get_mut_slot(instr.a1()) =
                    Value::Bool(Value::op_not(*frame.get_slot(instr.a2())));
            }

            InstrKind::CmpEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_equal(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpNotEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_not_equal(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpGreater => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_greater(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpLess => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_less(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpGreaterOrEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_greater_or_equal(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
                ));
            }

            InstrKind::CmpLessOrEqual => {
                *frame.get_mut_slot(instr.a1()) = Value::Bool(Value::cmp_less_or_equal(
                    *frame.get_slot(instr.a2()),
                    *frame.get_slot(instr.a3()),
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

            InstrKind::JumpNotIf => {
                if !frame.get_slot(instr.a2()).is_truthy() {
                    frame.ip = instr.a1();
                }
            }

            InstrKind::PackTuple => {
                let tuple: Box<[Value<'gc>]> = (&frame.stack_slice()
                    [instr.a2() as usize..(instr.a2() + instr.a3()) as usize])
                    .into();
                *frame.get_mut_slot(instr.a1()) = Value::Tuple(Gc::new(mc, tuple));
            }

            InstrKind::UnpackTuple => {
                let Value::Tuple(tuple) = frame.get_slot(instr.a2()) else {
                    panic!("Trying to unpack non-tuple");
                };
                let tuple = Gc::clone(tuple);

                if instr.a3() as usize != tuple.len() {
                    panic!("Tuple len not match.");
                }

                frame.stack_slice_mut()[instr.a1() as usize..(instr.a1() + instr.a3()) as usize]
                    .clone_from_slice(&tuple);
            }

            InstrKind::PackArray => {
                let slot = instr.a2();
                let count = instr.a3();

                let mut array = Vec::with_capacity(count as usize);

                for i in 0..count {
                    array.push(*frame.get_slot(slot + i));
                }

                *frame.get_mut_slot(instr.a1()) = Value::Array(Gc::new(mc, RefLock::new(array)));
            }

            // InstrKind::PackDict => {
            //     let slot = instr.a2();
            //     let count = instr.a3();

            //     let mut dict: HashMap<Gc<Box<str>>, Value<'gc>> = HashMap::new();

            //     for i in 0..count {
            //         let Value::String(key) = frame.get_slot(slot + 2 * i) else {
            //             panic!("Key must be string.");
            //         };
            //         let Value<'gc> = frame.get_slot(slot + 2 * i + 1).clone();

            //         dict.insert(Gc::clone(key), Value<'gc>);
            //     }

            //     *frame.get_mut_slot(instr.a1()) = Value::Dict(Gc::new(GcCell::new(dict)));
            // }
            InstrKind::Call => {
                let Value::Callable(callable) = frame.get_slot(instr.a1()) else {
                    panic!("Trying to call non-callable.");
                };

                let args = &frame.stack_slice()
                    [(instr.a2() + 1) as usize..(instr.a2() + 1 + instr.a3()) as usize];

                match callable {
                    // If we are calling a native function, we don't need to create a call frame
                    Callable::Native(NativeFunc(native_func)) => {
                        *frame.get_mut_slot(instr.a2()) = native_func(args, mc);
                    }

                    Callable::Func(code_obj) => {
                        let code_obj_copy = Gc::clone(code_obj);

                        frame.ret_slot = instr.a2();

                        self.frames.push_frame_within(
                            code_obj_copy,
                            (instr.a2() + 1) as usize..(instr.a2() + 1 + instr.a3()) as usize,
                        );
                    }
                }
            }

            InstrKind::Return => {
                let ret_val = *frame.get_slot(instr.a1());

                self.frames.pop_frame();

                let old_frame = self.frames.get_last_mut();
                *old_frame.get_mut_slot(old_frame.ret_slot) = ret_val;
            }
        }
    }
}

pub struct VmHandle {
    arena: Arena<Rootable![Vm<'_>]>,
}

impl VmHandle {
    pub fn new(use_builtins: bool) -> Self {
        let arena = Arena::<Rootable![Vm<'_>]>::new(|_| Vm::new(use_builtins));

        Self { arena }
    }

    pub fn step(&mut self) {
        self.mutate_root(|mc, root| {
            root.step(mc);
        });
    }

    pub fn finish(&mut self) {
        loop {
            let finished = self.mutate_root(|mc, root| {
                root.step(mc);
                root.finished
            });

            if finished {
                break;
            }
        }
    }

    pub fn mutate<F, T>(&self, f: F) -> T
    where
        F: for<'gc> FnOnce(&'gc Mutation<'gc>, &'gc Vm<'gc>) -> T,
    {
        self.arena.mutate(f)
    }

    pub fn mutate_root<F, T>(&mut self, f: F) -> T
    where
        F: for<'gc> FnOnce(&'gc Mutation<'gc>, &'gc mut Vm<'gc>) -> T,
    {
        self.arena.mutate_root(f)
    }
}
