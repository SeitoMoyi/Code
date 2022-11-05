package chapter07;

public class VarParameter {
    public static void main(String[] args){
        Method method = new Method();
        int[] arr = {1,2,3};
        System.out.println(method.sum(arr));
    }
}
class Method{
    // 可变参数
    // 可变参数的实参可以为0个或任意多个。
    // 可变参数的实参课可以为数组。
    // 可变参数的本质就是数组。
    // 可变参数可以和普通类型的参数一起放在星灿烈表，但必须保证可变参数在最后。
    // 一个形参列表只能出现一个可变参数。
    public int sum(int... nums){
        System.out.println("数据长度=" + nums.length);
        int res = 0;
        for(int i = 0; i < nums.length; i++) res += nums[i];
        return res;
    }
}
