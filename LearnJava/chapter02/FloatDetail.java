package chapter02;

public class FloatDetail{
    public static void main(String[] args) {
        double num1 = 2.7; //2.7
        double num2 = 8.1/3; //2.6999999999999997
        System.out.println(num1);
        System.out.println(num2);
        //当我们对运算结果是小数的进行相等判断时，要小心
        //应该是以两个数的差值的绝对值，在某个精度范围内判断
        if (num1 == num2){
            System.out.println("相等");
        }
        if(Math.abs(num1 - num2) < 1e-8){
            System.out.println("match");
        }
    }
}