using Regression_analysis;
using System.Text;

class Vectors
{
    public (int, int) Shape;
    public int Size { get => Shape.Item1 * Shape.Item2; }
    private readonly double[][] values;
    private bool transposes = false;

    public Vectors(double[][] value) {
        this.values = value;
        this.Shape = (value.Length, value[0].Length);
    }
    public Vectors(double[] value) {
        this.values = [value];
        this.Shape = (1, value.Length);
    } 

    public static Vectors InitVectors((int, int) shape) {
        double[][] result = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++)
            result[i] = new double[shape.Item2];
        return new Vectors(result);
    }
    public static Vectors Zeros((int, int) shape) {
        double[][] temp_val = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++) {
            temp_val[i] = new double[shape.Item2];
            for (int j = 0; j < shape.Item2; j++)
                temp_val[i][j] = 0.0;
        }
        return new Vectors(temp_val);
    }
    public static Vectors Ones((int, int) shape)
    {
        double[][] temp_val = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++)
        {
            temp_val[i] = new double[shape.Item2];
            for (int j = 0; j < shape.Item2; j++)
                temp_val[i][j] = 1.0;
        }
        return new Vectors(temp_val);
    }
    public static Vectors Eig((int, int) shape, double value = 1.0)
    {
        double[][] result = new double[shape.Item1][];
        for (int i = 0; i < shape.Item1; i++) {
            result[i] = new double [shape.Item2];
            for (int j = 0; j < i; j++)
                result[i][j] = 0.0;
            result[i][i] = value;
            for (int k = i + 1; k < shape.Item2; k++)
                result[i][k] = 0.0;
        }
        return new Vectors(result);
    }

    public Vectors Copy() {
        double[][] result = new double[this.Shape.Item1][];
        for (int i = 0; i < this.Shape.Item1; i++)
        {
            result[i] = new double[this.Shape.Item2];
            for (int j = 0; j < this.Shape.Item2; j++)
                result[i][j] = this[i, j];
        }
        return new Vectors(result);
    }

    public Vectors T() {
        Vectors result = this.Copy();
        result.transposes = !result.transposes;
        (result.Shape.Item1, result.Shape.Item2) = (result.Shape.Item2, result.Shape.Item1);
        return result;
    }
    public static Vectors Multipty(Vectors vec, double val) {
        if (double.Abs(val) < double.Epsilon) return Vectors.Zeros(vec.Shape);
        else {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] * val;
            }
            return new Vectors(result);
        }
    }
    public static Vectors Multipty(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] * val[i, j];
            }
            return new Vectors(result);
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Add(Vectors vec, double val)
    {
        if (double.Abs(val) >= double.Epsilon)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] + val;
            }
            return new Vectors(result);
        }
        return vec;
    }
    public static Vectors Add(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] + val[i, j];
            }
            return new Vectors(result);
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Sub(Vectors vec, double val) {
        if (double.Abs(val) >= double.Epsilon)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] - val;
            }
            return new Vectors(result);
        }
        return vec;
    }
    public static Vectors Sub(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] - val[i, j];
            }
            return new Vectors(result);
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public static Vectors Divade(Vectors vec, double val)
    {
        if (double.Abs(val) >= double.Epsilon)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                    result[i][j] = vec[i, j] / val;
            }
            return new Vectors(result);
        }
        else throw new Exception("Divade by Zero");
    }
    public static Vectors Divade(Vectors vec, Vectors val)
    {
        if (vec.Shape == val.Shape)
        {
            double[][] result = new double[vec.Shape.Item1][];
            for (int i = 0; i < vec.Shape.Item1; i++)
            {
                result[i] = new double [vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                {
                    if (double.Abs(val[i, j]) < double.Epsilon) throw new Exception("Divade by Zero");
                    result[i][j] = vec[i, j] / val[i, j];
                }
            }
            return new Vectors(result);
        }
        else throw new Exception($"The shapes of the vectors do not match. The shape of the first vector is {vec.Shape}. The shape of the second vector is {val.Shape}.");
    }
    public Vectors Dot(Vectors vec) {
        if (this.Shape.Item2 == vec.Shape.Item1)
        {
            double[][] result = new double[this.Shape.Item1][];
            for (int i = 0; i < this.Shape.Item1; i++)
            {
                result[i] = new double[vec.Shape.Item2];
                for (int j = 0; j < vec.Shape.Item2; j++)
                {
                    result[i][j] = 0.0;
                    for (int k = 0; k < this.Shape.Item2; k++)
                        result[i][j] += this[i, k] * vec[k, j];
                }
            }
            return new Vectors(result);
        }
        else throw new Exception($"Vector multiplication requires the shape to be (n, m) -> (m, k). Currently it is {this.Shape} -> {vec.Shape}.");
    }
    public Vectors Square() {
        if (this.Shape.Item1 == this.Shape.Item2) {
            double[][] result = new double[this.Shape.Item1][];
            for (int i = 0; i < this.Shape.Item1; i++)
            {
                result[i] = new double[this.Shape.Item2];
                for (int j = 0; j < this.Shape.Item2; j++)
                {
                    result[i][j] = 0;
                    for (int k = 0; k < this.Shape.Item2; k++)
                        result[i][j] += this[i, k] * this[k, j];
                }
            }
            return new Vectors(result);
        }
        else throw new Exception($"Vector multiplication requires the shape to be (n, m) -> (m, k). Currently it is {this.Shape} -> {this.Shape}.");
    }
    public static double Max(Vectors vec) {
        double result = 0;
        for (int i = 0; i < vec.Shape.Item1; i++)
            for (int j = 0; j < vec.Shape.Item2; j++) {
                var tmp = double.Abs(vec[i, j]);
                if (tmp > result)
                    result = tmp;
            }
        return result;
    }
    public static double Min(Vectors vec)
    {
        double result = double.MaxValue;
        for (int i = 0; i < vec.Shape.Item1; i++)
            for (int j = 0; j < vec.Shape.Item2; j++)
            {
                var tmp = double.Abs(vec[i, j]);
                if (tmp < result)
                    result = tmp;
            }
        return result;
    }
    public static Vectors Inv_Schulz(Vectors A, int max_iter = 100) {
        if (A.Shape.Item1 != A.Shape.Item2)
            throw new Exception("The Vector in not square. Cannon be reversed.");
        var A_T = A.T();
        var A_inv = Vectors.Divade(A_T, Vectors.Norm(A.Dot(A_T)));
        var E_2 = Vectors.Eig(A.Shape, 2.0);
        for (int i = 0; i < max_iter; i++) {
            A_inv = A_inv.Dot(Vectors.Sub(E_2, A.Dot(A_inv)));
        }
        return A_inv;
    }
    public static double Norm(Vectors vec) {
        double summator = 0;
        for (int i = 0; i < vec.Shape.Item1; i++)
            for (int j = 0; j < vec.Shape.Item2; j++)
                summator += vec[i, j] * vec[i, j];
        return Math.Sqrt(summator);
    }

    public static double Power_Iteration(Vectors A, double tol = 1e-1, int max_iter = 1000) {
        var tmp = new Random_distribution();
        Vectors b_k = tmp.Uniform(0, 1, (1, A.Shape.Item1)).T();
        double eigenvalue_old = 0.0, eigenvalue_new = 0.0;
        Vectors bk_1, bk_T;
        for (int i = 0; i < max_iter; i++) {
            bk_1 = A.Dot(b_k);
            b_k = Vectors.Divade(bk_1, Vectors.Norm(bk_1));
            bk_T = b_k.T();
            eigenvalue_new = bk_T.Dot(A.Dot(b_k))[0, 0] / bk_T.Dot(b_k)[0, 0];
            if (double.Abs(eigenvalue_new - eigenvalue_old) < tol) break;
            eigenvalue_old = eigenvalue_new;
        }
        return eigenvalue_new;

    }

    public static Vectors Inv(Vectors A) {
        if (A.Shape.Item1 == A.Shape.Item2)
        {
            Vectors L = Vectors.InitVectors(A.Shape);
            Vectors U = Vectors.InitVectors(A.Shape);
            // LU-разложение
            for (int i = 0; i < A.Shape.Item1; i++)
            {
                for (int j = i; j < A.Shape.Item2; j++)
                {
                    U.values[i][j] = A.values[i][j];
                    for (int k = 0; k < i; k++)
                    {
                        U.values[i][j] -= L.values[i][k] * U.values[k][j];
                    }
                }
                for (int j = i; j < A.Shape.Item1; j++)
                {
                    if (i == j)
                    {
                        L.values[i][i] = 1; // диагональные элементы равны 1
                    }
                    else
                    {
                        L.values[j][i] = A.values[j][i];
                        for (int k = 0; k < i; k++)
                        {
                            L.values[j][i] -= L.values[j][k] * U.values[k][i];
                        }
                        L.values[j][i] /= U.values[i][i];
                    }
                }
            }

            // Теперь решим систему уравнений для получения обратной матрицы
            Vectors A_inv = Vectors.InitVectors(A.Shape);

            for (int i = 0; i < A.Shape.Item1; i++)
            {
                // Решаем Ly = e_i для y
                double[] y = new double[A.Shape.Item1];
                y[i] = 1;
                for (int j = 0; j < A.Shape.Item2; j++)
                {
                    for (int k = 0; k < j; k++)
                    {
                        y[j] -= L.values[j][k] * y[k];
                    }
                }

                // Теперь решаем Ux = y для x
                for (int j = A.Shape.Item2 - 1; j >= 0; j--)
                {
                    A_inv.values[j][i] = y[j];
                    for (int k = j + 1; k < A.Shape.Item1; k++)
                    {
                        A_inv.values[j][i] -= U.values[j][k] * A_inv.values[k][i];
                    }
                    A_inv.values[j][i] /= U.values[j][j];
                }
            }

            return A_inv;
        }
        else throw new Exception("The Vector in not square. Cannon be reversed.");
    }
    public static Vectors GetRow(Vectors vector, int index) {
        double[] Result = new double[vector.Shape.Item2];
        index = vector.GetIndex(index, false);
        if (vector.transposes)
        {
            for (int j = 0; j < vector.Shape.Item2; j++)
                Result[j] = vector.values[j][index];
        }
        else {
            for (int i = 0; i < vector.Shape.Item2; i++)
                Result[i] = vector.values[index][i];
        }
        return new Vectors(Result);
    }
    public void SetRow(Vectors row, int index) {
        if (!row.IsVector())
            throw new Exception("row is not Vector");
        if (row.Size != this.Shape.Item2)
            throw new Exception("The 'row' size does not match the vector row length");
        index = this.GetIndex(index, false);
        if (this.transposes)
        {
            for (int j = 0; j < this.Shape.Item2; j++)
                this.values[j][index] = row[j];
        }
        else
        {
            for (int i = 0; i < this.Shape.Item2; i++)
                this.values[index][i] = row[i];
        }
    }
    private int GetIndex(int index, bool Item2 = true) {
        if (Item2)
        {
            if (int.Abs(index) > Shape.Item2)
                throw new Exception("Index out of range");
            if (index < 0)
                return index + Shape.Item2;
            return index;
        }
        else {
            if (int.Abs(index) > Shape.Item1)
                throw new Exception("Index out if range.");
            if (index < 0)
                return index + Shape.Item1;
            return index;
        }
    }
    public bool IsVector() {
        if (Shape.Item1 == 1 || Shape.Item2 == 1) return true;
        else return false;
    }
    
    public override string? ToString() {
        if (this.Size > 0)
        {
            StringBuilder sb = new();
            for (int i = 0; i < this.Shape.Item1; i++)
            {
                sb.Append($"[ {this[i, 0].ToString("E2")}");
                for (int j = 1; j < this.Shape.Item2; j++)
                {
                    sb.Append($" {this[i, j].ToString("E2")}");
                }
                sb.Append(" ]\n");
            }
            return sb.ToString();
        }
        else return "Empty";
    }
    public double this[int i, int j] {
        get {
            int tmp = GetIndex(i, false), tmp_2 = GetIndex(j);
            if (transposes)
                return values[tmp_2][tmp];
            else
                return values[tmp][tmp_2];
        }
        set {
            int tmp = GetIndex(i, false), tmp_2 = GetIndex(j);
            if (transposes)
                values[tmp_2][tmp] = value;
            else
                values[tmp][tmp_2] = value;
        }
    }
    public double this[int i] {
        get {
            if (transposes) {
                int tmp = GetIndex(i, false);
                if (Shape.Item1 == 1)
                    return values[tmp][0];
                else
                    return values[0][tmp];
            }
            else {
                int tmp = GetIndex(i);
                if (Shape.Item1 == 1)
                    return values[0][tmp];
                else
                    return values[tmp][0];
            }
        }
        set {
            if (transposes) {
                int tmp = GetIndex(i, false);
                if (Shape.Item1 == 1)
                    values[tmp][0] = value;
                else
                    values[0][tmp] = value;
            }
            else {
                int tmp = GetIndex(i);
                if (Shape.Item1 == 1)
                    values[0][tmp] = value;
                else
                    values[tmp][0] = value;
            }
        }
    }
    
    public static Vectors operator +(Vectors s1, Vectors s2) => Add(s1, s2);
    public static Vectors operator +(Vectors s1, double s2) => Add(s1, s2);
    public static Vectors operator +(double s1, Vectors s2) => Add(s2, s1);
    public static Vectors operator -(Vectors s1, Vectors s2) => Sub(s1, s2);
    public static Vectors operator -(Vectors s1, double s2) => Sub(s1, s2);
    public static Vectors operator -(double s1, Vectors s2) => Sub(s2, s1);
    public static Vectors operator *(Vectors s1, Vectors s2) => Multipty(s1, s2);
    public static Vectors operator *(Vectors s1, double s2) => Multipty(s1, s2);
    public static Vectors operator *(double s1, Vectors s2) => Multipty(s2, s1);
    public static Vectors operator /(Vectors s1, Vectors s2) => Divade(s1, s2);
    public static Vectors operator /(Vectors s1, double s2) => Divade(s1, s2);
    public static Vectors operator /(double s1, Vectors s2) => Divade(s2, s1);
    public static Vectors operator &(Vectors s1, Vectors s2) => s1.Dot(s2);
    public static Vectors operator -(Vectors s1) => Multipty(s1, -1);
}



