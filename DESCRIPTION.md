## precedence of operator from lower to higher
- Equality
- Comparison
- Term
- Factor
- Unary

## Program rules
program -> statement* EOF ;

statement -> exprStmt | printStmt ;

exprStmt -> expression ;

expression -> "print" expression ;
