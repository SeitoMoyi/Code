package chapter07;

import java.util.Scanner;

public class Exercise {
    public static void main(String[] args){
        A a = new A();
        Scanner scan = new Scanner(System.in);
        double num = scan.nextDouble();
        scan.close();
        if(a.att(num) != null)
            System.out.println(a.att(num));
        else System.out.println("输入不能为0");
    }
}
class A{
    // 大写的Double作为包装类可以返回null，小写的double只能返回浮点数。
    public Double att(double num){
        if(num == 0) return null;
        else return num;
    }
}
