package chapter07;

public class EightQueen {
    public static void main(String[] args){
        Chess chess = new Chess();
        int[] map = new int[8];
        chess.queen(map, 0);
        System.out.println("一共有 " + chess.cnt + " 种");
    }
}
class Chess{
    public int cnt = 0;
    public void queen(int[] map, int n){
        if(n == 8) {
            for(int i = 0; i < 8; i ++){
                System.out.print(map[i] + " ");
            }
            System.out.println();
            cnt++;
        }
        else if(n == 0){
            for(int i = 0; i < 8; i ++){
                map[n] = i;
                queen(map, n+1);
            }
        }
        else {
            for(int i = 0; i < 8; i ++){
                if(check(map, n, i)){
                    map[n] = i;
                    queen(map, n+1);
                }
            }
        }
    }
    public boolean check(int[] map, int n, int i){
        for(int j = 0; j < n; j++){
            if (j - map[j] == n - i || j + map[j] == n + i || map[j] == i) return false;
        }
        return true;
    }
}
