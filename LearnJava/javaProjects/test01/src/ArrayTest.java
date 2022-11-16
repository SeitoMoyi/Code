public class ArrayTest {
    public static void main(String[] args) {
        MyTools myTools = new MyTools();
        int[] arr = {1,3,5,6,4};
        myTools.bubble(arr);
        for (int i = 0; i < arr.length; i++) {
            System.out.print(arr[i] + "\t");
        }
    }
}
// 创建一个冒泡排序

class MyTools{
    public void bubble(int[] arr){
        int temp;
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]){
                    temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}