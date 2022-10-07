package chapter03;

public class BasicToString {
    public static void main(String[] args){
        int n1 = 100;
        float f1 = 100.0f;
        boolean b1 = true;
        String s1 = n1 + "";
        String s2 = f1 + "";
        String s3 = b1 + "";
        System.out.println(s1 + " " + s2 + " " + s3);
    }
}
