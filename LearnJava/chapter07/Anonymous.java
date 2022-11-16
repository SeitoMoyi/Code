package chapter07;

public class Anonymous {
    int count = 9;
    public void count1(){
        count = 10;
        System.out.println("count1 = " + count);
    }
    public void count2(){
        System.out.println("count2 = " + count++);
    }
    public static void main(String[] args){
        new Anonymous().count1();// 匿名对象
        // 1.匿名对象使用后就不能使用
        // 2.创建好匿名对象后就立即调用方法
        Anonymous a = new Anonymous();
        a.count2();
        a.count2();
    }
}
