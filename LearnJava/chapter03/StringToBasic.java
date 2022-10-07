package chapter03;

public class StringToBasic {
    public static void main(String[] args){
        String s1 = "123";
        int num1 = Integer.parseInt(s1);
        double num2 = Double.parseDouble(s1);
        float num3 = Float.parseFloat(s1);
        byte num4 = Byte.parseByte(s1);
        boolean num5 = Boolean.parseBoolean("true");
        System.out.println(num1 + " " + num2 + " " + num3 + " " + num4 + " " + num5);

        System.out.println(s1.charAt(0));//取第一个字符
    }
}