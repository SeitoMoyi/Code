package chapter07;

public class OverLoad {
    public static void main(String[] args){
        MyCalc mycalc = new MyCalc();
        System.out.println(mycalc.calc(2, 1.3));
    }
}
class MyCalc{
    // 方法重载,同样的方法名是合法的
    // 方法名：相同；
    // 形参：类型或顺序至少有一样不同；
    // 返回类型：无要求
    public int calc(int n1, int n2){
        return n1 + n2;
    }
    public double calc(int n1, double n2){
        return n1 + n2;
    }
    public double calc(double n1, int n2){
        return n1 + n2;
    }
    public int calc(int n1, int n2, int n3){
        return n1 + n2 + n3;
    }
}