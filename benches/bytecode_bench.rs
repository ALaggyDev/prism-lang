use criterion::{criterion_group, criterion_main, Criterion};
use gc::Gc;
use prism_lang::{
    bytecode::{Callable, CodeObject, Value, Vm},
    compiler::compile,
    instr, stage_1, stage_2,
};
use std::hint::black_box;

fn main_wrapper(fib: Gc<CodeObject>, n: f64) -> Gc<CodeObject> {
    Gc::new(CodeObject {
        code: Box::new([
            instr!(LoadConst, 0, 0),
            instr!(StoreGlobal, 0, 0),
            instr!(LoadConst, 2, 1),
            instr!(Call, 0, 1, 1),
            instr!(Return, 1),
        ]),
        consts: Box::new([Value::Callable(Callable::Func(fib)), Value::Number(n)]),
        global_names: Box::new(["fib".into()]),
        stack_count: 3,
        arg_count: 0,
    })
}

fn fib_recursive_normal() -> Gc<CodeObject> {
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
        stack_count: 8,
        arg_count: 1,
    });

    main_wrapper(fib, 25.0)
}

fn fib_iterative_normal() -> Gc<CodeObject> {
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
        global_names: Box::new([]),
        stack_count: 7,
        arg_count: 1,
    });

    main_wrapper(fib, 100.0)
}

fn fib_recursive_compiled() -> Gc<CodeObject> {
    let content = r#"
    fn fib(n) {
        if (n <= 1) {
            return n;
        }

        return fib(n - 1) + fib(n - 2);
    }

    fib(25);
    "#;

    let (tokens, interner) = stage_1(&content);
    let program = stage_2(&tokens).unwrap();

    Gc::new(compile(&program, &interner).unwrap())
}

fn fib_iterative_compiled() -> Gc<CodeObject> {
    let content = r#"
    fn fib(n) {
        let a = 0;
        let b = 1;
        let i = 2;
        while (i <= n) {
            let c = a + b;
            a = b;
            b = c;
            i = i + 1;
        }
        return b;
    }

    fib(100);
    "#;

    let (tokens, interner) = stage_1(&content);
    let program = stage_2(&tokens).unwrap();

    Gc::new(compile(&program, &interner).unwrap())
}

fn run(code_object: Gc<CodeObject>) {
    let mut vm = Vm::new_from_code_object(Gc::clone(&code_object), &[], false);

    while vm.result.is_none() {
        vm.step();
    }

    black_box(vm.result);
}

fn fib_benchmark(c: &mut Criterion) {
    let fib_recu_normal = fib_recursive_normal();
    let fib_iter_normal = fib_iterative_normal();
    let fib_recu_compiled = fib_recursive_compiled();
    let fib_iter_compiled = fib_iterative_compiled();

    c.bench_function("fib recursive 25", |b| {
        b.iter(|| run(Gc::clone(&fib_recu_normal)))
    });
    c.bench_function("fib iterative 100", |b| {
        b.iter(|| run(Gc::clone(&fib_iter_normal)))
    });
    c.bench_function("fib recursive 25 compiled", |b| {
        b.iter(|| run(Gc::clone(&fib_recu_compiled)))
    });
    c.bench_function("fib iterative 100 compiled", |b| {
        b.iter(|| run(Gc::clone(&fib_iter_compiled)))
    });
}

criterion_group!(benches, fib_benchmark);
criterion_main!(benches);
