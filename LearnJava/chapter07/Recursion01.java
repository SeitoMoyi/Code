package chapter07;

public class Recursion01 {
    public static void main(String[] args){
        int[][] map = new int[8][8];
        for(int i = 0; i < map[0].length; i++){
            map[0][i] = 1;
            map[7][i] = 1;
        }
        for(int i = 0; i < map.length; i++){
            map[i][0] = 1;
            map[i][7] = 1;
        }
        map[2][1] = 1;
        map[2][2] = 1;
        map[3][3] = 1;
        for(int i = 0; i < map.length; i++){
            for(int j = 0; j < map[i].length; j++){
                System.out.print(map[i][j] + " ");
            }
            System.out.println();
        }
        Tool t1 = new Tool();
        t1.findWay(map, 1, 1);
        System.out.println("=============");
        for(int i = 0; i < map.length; i++){
            for(int j = 0; j < map[i].length; j++){
                System.out.print(map[i][j] + " ");
            }
            System.out.println();
        }
    }
}

class Tool{
    public boolean findWay(int[][] map, int i, int j){
        if(map[6][6] == 2) return true; 
        // 2表示可到达的,3表示走过但走不通
        else{
            if(map[i][j] == 0){
                map[i][j] = 2;
                if(findWay(map, i+1, j)) return true;
                else if(findWay(map, i, j+1)) return true;
                else if(findWay(map, i-1, j)) return true;
                else if(findWay(map, i-1, j-1)) return true;
                else {
                    map[i][j] = 3;
                    return false;
                }
            }
        }
        return false;
    }
}
