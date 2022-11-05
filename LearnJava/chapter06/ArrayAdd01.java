package chapter06;

import java.util.Scanner;

public class ArrayAdd01 {
    public static void main(String[] args){
        int[] arr = {1,2,3};
        Scanner myScanner = new Scanner(System.in);
        do{
            int[] arrnew = new int[arr.length+1];
            for (int i = 0; i < arr.length; i++)
            arrnew[i] = arr[i];
            arr = arrnew;
            System.out.println("请输入要加入的元素：");
            int addNum = myScanner.nextInt();
            arrnew[arrnew.length-1] = addNum;
            // 扩容数组：新建一个数组后复制给原数组
            System.out.println(arr.length);
            for (int i = 0; i < arr.length; i ++)
            System.out.print(arr[i]);
            System.out.println();
            System.out.println("是否继续添加：");
            char c = myScanner.next().charAt(0);
            if (c == 'n') break;
        }while(true);
        myScanner.close();
    }
}
