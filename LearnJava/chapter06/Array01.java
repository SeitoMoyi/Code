package chapter06;

import java.util.Scanner;

public class Array01 {
    public static void main(String[] args) {
        // char str[] = new char[5];
        // char str[] = {'s','h','a','b','i'};
        char str[];
        str = new char[5];
        Scanner myScanner = new Scanner(System.in);
        for(int i = 0; i < str.length; i ++){
            System.out.println("请输入第" + (i+1) + "个字符的值");
            str[i] = myScanner.next().charAt(0);
        }
        System.out.println(new String(str));
        myScanner.close();
    }
}