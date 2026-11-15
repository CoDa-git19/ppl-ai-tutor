These short notes on Principles of Programming Languages ship with the project so it runs before you add your own course notes.

## Lexical analysis

A lexer (scanner) turns a stream of characters into a stream of tokens such as identifiers, keywords, literals and operators. Token patterns are written as regular expressions and implemented with finite automata: a regular expression is converted to an NFA (Thompson's construction), then to a DFA (subset construction), and the DFA can be minimized.

Lexers usually follow the longest-match rule (maximal munch): read as many characters as possible for one token, so `>=` is one token, not `>` followed by `=`. When two patterns match the same length, the pattern listed first wins, which is why keywords are listed before the identifier rule.

```
ID      : [a-zA-Z_][a-zA-Z0-9_]* ;
INTLIT  : [0-9]+ ;
WS      : [ \t\r\n]+ -> skip ;
```

## Context-free grammars and parsing

A context-free grammar has terminals, nonterminals, productions and a start symbol. A grammar is ambiguous when some sentence has two different parse trees; the classic example is `E -> E + E | E * E | id`, which can be fixed by encoding precedence and associativity in separate nonterminals (expression, term, factor).

### LL(1) parsing

An LL(1) parser reads input Left to right, builds a Leftmost derivation and uses 1 token of lookahead. It needs a grammar with no left recursion and no common prefixes; left recursion `A -> A a | b` is rewritten as `A -> b A'` and `A' -> a A' | ε`.

FIRST(X) is the set of terminals that can begin a string derived from X. FOLLOW(A) is the set of terminals that can appear immediately after A in some sentential form; the end marker `$` is in FOLLOW of the start symbol. The parse table entry M[A, a] holds the production A -> α when a ∈ FIRST(α), or when α can derive ε and a ∈ FOLLOW(A). A grammar is LL(1) exactly when no table cell holds two productions.

### LR parsing

LR parsers (SLR, LALR, canonical LR) read Left to right and build a Rightmost derivation in reverse, using shift and reduce actions driven by a DFA of items. They accept a larger class of grammars than LL(1), including left-recursive ones. A shift/reduce conflict means the parser cannot decide whether to shift the next token or reduce; the dangling-else problem is the textbook case.

## Names, binding and scope

A binding associates a name with an attribute such as a type, a value or a memory location. Binding time can be language design time, compile time, load time or run time. Static binding happens before run time and does not change; dynamic binding happens or can change during execution.

The lifetime of a variable is the time during which it is bound to a storage location. Static variables live for the whole program, stack-dynamic variables live while their subprogram is active, and heap-dynamic variables are allocated and freed explicitly or by a garbage collector.

### Static scope và dynamic scope

Phạm vi tĩnh (static scope, lexical scope) xác định một tên tham chiếu tới khai báo nào dựa vào cấu trúc văn bản của chương trình: trình biên dịch tìm khai báo ở khối hiện tại, rồi lần lượt ra các khối bao ngoài. Phạm vi động (dynamic scope) dựa vào chuỗi lời gọi hàm lúc chạy: tên được tìm trong hàm đang gọi, rồi tới hàm đã gọi nó.

```
x = 1
def f(): return x
def g():
    x = 2
    return f()
# static scope: g() trả về 1 ; dynamic scope: g() trả về 2
```

Phần lớn ngôn ngữ hiện đại (C, Java, Python, Scala) dùng phạm vi tĩnh vì dễ đọc và cho phép kiểm tra lúc biên dịch. Một số phiên bản Lisp cũ và biến đặc biệt trong Common Lisp dùng phạm vi động.

## Type systems

A type system assigns types to expressions and rejects programs that would apply an operation to values of the wrong type. Static typing checks types at compile time (Java, Scala, Haskell); dynamic typing checks types at run time (Python, JavaScript).

Strong versus weak typing is a separate question from static versus dynamic: a strongly typed language does not silently reinterpret a value as an unrelated type, while a weakly typed language performs many implicit conversions. Python is dynamically but strongly typed (`"1" + 1` raises an error); C is statically but fairly weakly typed (pointer casts, implicit integer conversions).

Type coercion is an implicit conversion inserted by the compiler, for example int to float in `1 + 2.5`. Type inference lets the compiler deduce types without annotations; Hindley–Milner inference in ML and Haskell finds the most general type of an expression.

## Subprograms and parameter passing

In pass-by-value the callee receives a copy, so assignments inside the callee do not affect the caller. In pass-by-reference the callee receives an alias of the caller's variable, so assignments are visible to the caller. Pass-by-value-result copies in at the call and copies back at the return. Pass-by-name re-evaluates the argument expression every time the parameter is used, as in Algol 60 and Scala's `=> T` parameters.

Activation records (stack frames) store parameters, local variables, the return address, the dynamic link to the caller's frame and, in languages with nested functions, the static link to the frame of the enclosing function, which implements static scope.

## Functional programming

Functions are first-class values: they can be passed as arguments, returned as results and stored in data structures. A higher-order function takes or returns functions; `map`, `filter` and `foldLeft` are the standard examples.

A closure is a function together with the environment in which it was defined, so it can use variables of the enclosing scope even after that scope has returned.

```scala
def makeAdder(n: Int): Int => Int = (x: Int) => x + n
val add5 = makeAdder(5)
add5(10)   // 15 : the closure remembers n = 5
```

Pure functions have no side effects and always return the same result for the same arguments, which makes reasoning, testing and parallel evaluation easier. Recursion replaces loops; a tail-recursive call is the last action of the function and can be compiled into a loop.

## Object-oriented programming

Encapsulation hides the representation of an object behind an interface. Inheritance lets a subclass reuse and extend a superclass. Subtype polymorphism allows a value of a subclass to be used wherever the superclass is expected.

Dynamic dispatch selects the method implementation at run time from the actual class of the receiver object, usually through a virtual method table (vtable). Static dispatch, by contrast, selects the method at compile time from the declared type, as with non-virtual methods in C++.
