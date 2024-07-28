use criterion::{criterion_group, criterion_main, Criterion};
use gc::Gc;
use prism_lang::{
    bytecode::{CallFrame, Callable, CodeObject, Value, Vm},
    instr,
    native_func::NATIVE_FUNCS,
};
use std::{collections::HashMap, hint::black_box};

fn fib_recursive(n: f64) {
    let fib = Gc::new(CodeObject {
        code: Box::new([
            // load 1 & 2
            instr!(LoadConst, 1, 0),
            instr!(LoadConst, 2, 1),
            // if n <= 1, return n
            instr!(CmpGreater, 3, 0, 1),
            instr!(JumpIf, 5, 3),
            instr!(Return, 0),
            // load fib
            instr!(LoadGlobal, 3, 0),
            // call fib(n - 1)
            instr!(OpMinus, 5, 0, 1),
            instr!(Call, 3, 4, 1),
            // call fib(n - 2)
            instr!(OpMinus, 7, 0, 2),
            instr!(Call, 3, 6, 1),
            // return fib(n - 1) + fib(n - 2)
            instr!(OpAdd, 0, 4, 6),
            instr!(Return, 0),
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
        code: Box::new([
            // 1: a, 2: b, 3: c, 4: i
            instr!(LoadConst, 1, 0),
            instr!(LoadConst, 2, 1),
            instr!(LoadConst, 4, 2),
            instr!(LoadConst, 5, 1),
            // c = a + b
            // a = b
            // b = c
            instr!(OpAdd, 3, 1, 2),
            instr!(Copy, 1, 2),
            instr!(Copy, 2, 3),
            // i++
            // if i <= n, loop again
            instr!(OpAdd, 4, 4, 5),
            instr!(CmpLessOrEqual, 6, 4, 0),
            instr!(JumpIf, 4, 6),
            // return b
            instr!(Return, 2),
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
