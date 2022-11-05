package chapter06;
public class Array02 {
    public static void main(String[] args){
        int[] arr1 = {4,-1,9,10,23};
        int[] arr2 = arr1;

        // 数组的赋值是地址拷贝
        // int[] arr2 = new int[arr1.length];

        arr2[0] = 0;
        for(int i = 0 ; i < arr1.length ; i ++){
            System.out.print(arr1[i]);
        }
        System.out.println();
    }
}
