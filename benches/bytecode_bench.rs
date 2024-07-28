use criterion::{criterion_group, criterion_main, Criterion};
use gc::Gc;
use prism_lang::{
    bytecode::{CallFrame, Callable, CodeObject, Instr, Value, Vm},
    native_func::NATIVE_FUNCS,
};
use std::{collections::HashMap, hint::black_box};

fn fib_recursive(n: f64) {
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

    let call_frame = CallFrame::new(Gc::clone(&fib), &[Value::Number(n)]);

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

    black_box(vm.result);
}

fn fib_iterative(n: f64) {
    let fib = Gc::new(CodeObject {
        #[rustfmt::skip]
        code: Box::new([
            // 1: a, 2: b, 3: c, 4: i

            Instr::LoadConst { dest: 1, index: 0 },
            Instr::LoadConst { dest: 2, index: 1 },
            Instr::LoadConst { dest: 4, index: 2 },
            Instr::LoadConst { dest: 5, index: 1 },

            // c = a + b
            // a = b
            // b = c
            Instr::OpAdd { dest: 3, op1: 1, op2: 2 },
            Instr::Copy { dest: 1, src: 2 },
            Instr::Copy { dest: 2, src: 3 },

            // i++
            // if i <= n, loop again
            Instr::OpAdd { dest: 4, op1: 4, op2: 5 },
            Instr::CmpLessOrEqual { dest: 6, op1: 4, op2: 0 },
            Instr::JumpIf { dest: 4, op: 6 },

            // return b
            Instr::Return { src: 2 },
        ]),
        consts: Box::new([Value::Number(1.0), Value::Number(1.0), Value::Number(2.0)]),
        global_names: Box::new(["fib".into()]),
        name: "fib".into(),
        stack_count: 7,
        arg_count: 1,
    });

    let call_frame = CallFrame::new(Gc::clone(&fib), &[Value::Number(n)]);

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

    black_box(vm.result);
}

pub fn fib_benchmark(c: &mut Criterion) {
    c.bench_function("fib recursive 25", |b| b.iter(|| fib_recursive(25.0)));
    c.bench_function("fib iterative 100", |b| b.iter(|| fib_iterative(100.0)));
}

criterion_group!(benches, fib_benchmark);
criterion_main!(benches);
