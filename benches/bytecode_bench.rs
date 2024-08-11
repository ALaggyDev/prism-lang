use criterion::{criterion_group, criterion_main, Criterion};
use gc_arena::{Gc, Mutation};
use prism_lang::{
    compiler::compile,
    instr, lex, parse,
    vm::{Callable, CodeObject, Value, VmHandle},
};
use string_interner::StringInterner;

fn main_wrapper<'gc>(
    mc: &Mutation<'gc>,
    fib: Gc<'gc, CodeObject<'gc>>,
    n: f64,
) -> Gc<'gc, CodeObject<'gc>> {
    Gc::new(
        mc,
        CodeObject {
            code: Box::new([
                instr!(LoadConst, 0, 0),
                instr!(StoreGlobal, 0, 0),
                instr!(LoadConst, 2, 1),
                instr!(Call, 0, 1, 1),
            ]),
            consts: Box::new([Value::Callable(Callable::Func(fib)), Value::Number(n)]),
            global_names: Box::new(["fib".into()]),
            stack_count: 3,
            arg_count: 0,
        },
    )
}

fn fib_recursive_normal<'gc>(mc: &Mutation<'gc>) -> Gc<'gc, CodeObject<'gc>> {
    let fib = Gc::new(
        mc,
        CodeObject {
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
        },
    );

    main_wrapper(mc, fib, 25.0)
}

fn fib_iterative_normal<'gc>(mc: &Mutation<'gc>) -> Gc<'gc, CodeObject<'gc>> {
    let fib = Gc::new(
        mc,
        CodeObject {
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
        },
    );

    main_wrapper(mc, fib, 100.0)
}

fn fib_recursive_compiled<'gc>(mc: &'gc Mutation<'gc>) -> Gc<'gc, CodeObject<'gc>> {
    let content = r#"
    fn fib(n) {
        if (n <= 1) {
            return n;
        }

        return fib(n - 1) + fib(n - 2);
    }

    fib(25);
    "#;

    let mut interner = StringInterner::new();
    let tokens = lex(content, &mut interner).unwrap();
    let program = parse(&tokens, false).unwrap();

    Gc::new(mc, compile(mc, &program, &interner).unwrap())
}

fn fib_iterative_compiled<'gc>(mc: &'gc Mutation<'gc>) -> Gc<'gc, CodeObject<'gc>> {
    let content = r#"
    fn fib(n) {
        let a = 0;
        let b = 1;
        for i in 1..n {
		    let c = a + b;
		    a = b;
		    b = c;
	    }
        return b;
    }

    fib(100);
    "#;

    let mut interner = StringInterner::new();
    let tokens = lex(content, &mut interner).unwrap();
    let program = parse(&tokens, false).unwrap();

    Gc::new(mc, compile(mc, &program, &interner).unwrap())
}

fn run<F>(f: F)
where
    F: for<'gc> FnOnce(&'gc Mutation<'gc>) -> Gc<'gc, CodeObject<'gc>>,
{
    let mut handle = VmHandle::new(false);

    handle.mutate_root(|mc, root| {
        let code_object = f(mc);

        root.frames.push_frame(code_object, &[]);
    });

    handle.finish();
}

fn fib_benchmark(c: &mut Criterion) {
    c.bench_function("fib recursive 25", |b| b.iter(|| run(fib_recursive_normal)));
    c.bench_function("fib iterative 100", |b| {
        b.iter(|| run(fib_iterative_normal))
    });
    c.bench_function("fib recursive 25 compiled", |b| {
        b.iter(|| run(fib_recursive_compiled))
    });
    c.bench_function("fib iterative 100 compiled", |b| {
        b.iter(|| run(fib_iterative_compiled))
    });
}

criterion_group!(benches, fib_benchmark);
criterion_main!(benches);
