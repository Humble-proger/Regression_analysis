using Regression_analysis.Types;

namespace Regression_analysis.RandomDist
{
    class Random_distribution
    {
        private readonly Random rand;
        private readonly double[] gamma_constats = new double[8]; // 0 - m, 1 - s_2, 2 - s, 3 - d, 4 - b, 5 - w, 6 - v, 7 - c;
        private readonly double M2 = Math.Sqrt(2);
        private readonly double M3 = Math.Sqrt(3);

        public Random_distribution(int? seed = null) {
            if (seed == null)
                this.rand = new Random();
            else
                this.rand = new Random((int) seed);
        }
        public double Uniform(double a, double b) => a + this.rand.NextDouble() * (b - a);
        public Vectors Uniform(double a, double b, (int, int) Shape) {
            Vectors vec = Vectors.InitVectors(Shape);
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Shape.Item2; j++)
                    vec[i, j] = this.Uniform(a, b);
            return vec;
        }
        public double Exponential(double loc = 0.0, double scale = 1.0) => -Math.Log(1 - this.rand.NextDouble()) * scale + loc;
        public Vectors Exponential((int, int) Shape, double loc = 0.0, double scale = 1.0) {
            Vectors Result = Vectors.InitVectors(Shape);
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Shape.Item2; j++)
                    Result[i, j] = Exponential(loc, scale);
            return Result;
        }
        public double Laplace(double loc = 0.0, double scale = 1.0) => loc + double.Sign(Uniform(-1, 1)) * Exponential(0, scale);
        public Vectors Laplace((int, int) Shape, double loc = 0.0, double scale = 1.0)
        {
            Vectors Result = Vectors.InitVectors(Shape);
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Shape.Item2; j++)
                    Result[i, j] = Laplace(loc, scale);
            return Result;
        }
        public double Cauchy(double loc = 0.0, double scale = 1.0) => loc + scale * Math.Tan(double.Pi * (Uniform(-1, 1) - 0.5));
        public Vectors Cauchy((int, int) Shape, double loc = 0.0, double scale = 1.0) {
            Vectors Result = Vectors.InitVectors(Shape);
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Shape.Item2; j++)
                    Result[i, j] = Cauchy(loc, scale);
            return Result;
        }
        public double Normal(double loc = 0.0, double scale = 1.0) {
            double E1 = this.Uniform(-1, 1), E2 = this.Uniform(-1, 1);
            double s = E1 * E1 + E2 * E2;
            while (s > 1 || s < double.Epsilon) {
                E1 = this.Uniform(-1, 1); E2 = this.Uniform(-1, 1);
                s = E1 * E1 + E2 * E2;
            }
            return E1 * Math.Sqrt(-2 * Math.Log(s) / s) * scale + loc;
        }
        public Vectors Normal((int, int) Shape, double loc = 0.0, double scale = 1.0) {
            Vectors Result = Vectors.InitVectors(Shape);
            int Count_1 = Shape.Item1 / 2, Count_2 = Shape.Item2 / 2;
            double val;
            for (int i = 0; i < Shape.Item1; i++)
                for (int j = 0; j < Count_2; j++) {
                    double E1 = this.Uniform(-1, 1), E2 = this.Uniform(-1, 1);
                    double s = E1 * E1 + E2 * E2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        E1 = this.Uniform(-1, 1); E2 = this.Uniform(-1, 1);
                        s = E1 * E1 + E2 * E2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * scale;
                    Result[i, 2 * j] = E1 * val + loc;
                    Result[i, 2 * j + 1] = E2 * val + loc;
                }
            if (Shape.Item2 % 2 == 1) {
                for (int i = 0; i < Count_1; i++) {
                    double E1 = this.Uniform(-1, 1), E2 = this.Uniform(-1, 1);
                    double s = E1 * E1 + E2 * E2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        E1 = this.Uniform(-1, 1); E2 = this.Uniform(-1, 1);
                        s = E1 * E1 + E2 * E2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * scale;
                    Result[2 * i, -1] = E1 * val + loc;
                    Result[2 * i + 1, -1] = E2 * val + loc;
                }
                if (Shape.Item1 % 2 == 1) Result[-1, -1] = this.Normal(loc, scale);
            }
            return Result;
        }

        private double GA1(int k = 1) {
            double res = 0.0;
            for (int i = 0; i < k; i++)
                res += Exponential();
            return res;
        }
        private double GA2(double k = 0.5) {
            double res = Normal();
            int i = 1;
            while (i < k) {
                res += Exponential(); i++;
            }
            return res;
        }
        private double GS(double k) {
            double res, U, W;
            int iter = 0;
            do
            {
                U = Uniform(0, 1 + k / double.E);
                W = Exponential();
                if (U <= 1)
                {
                    res = Math.Pow(U, 1.0 / k);
                    if (res <= W)
                        return res;
                }
                else
                {
                    res = -Math.Log((1 - U) / k + 1.0 / double.E);
                    if ((1 - k) * Math.Log(res) <= W)
                        return res;
                }
            } while (++iter < 1e9);
            throw new Exception("Failed to calculate GS");
        }
        private double GF(double k) {
            double E1, E2;
            do
            {
                E1 = Exponential(); E2 = Exponential();
            } while (E2 < (k - 1) * (E1 - Math.Log(E1) - 1));
            return k * E1;
        }
        private void InitConstantGO(double k) {
            double val = Math.Sqrt(k);
            this.gamma_constats[0] = k - 1;
            this.gamma_constats[1] = 2 * this.M2 * val / this.M3  + k;
            this.gamma_constats[2] = Math.Sqrt(this.gamma_constats[1]);
            this.gamma_constats[3] = this.M2 * this.M3 * this.gamma_constats[1];
            this.gamma_constats[4] = this.gamma_constats[3] + this.gamma_constats[0];
            this.gamma_constats[5] = this.gamma_constats[1] / (this.gamma_constats[0] - 1);
            this.gamma_constats[6] = 2 * this.gamma_constats[1] / (this.gamma_constats[0] * val);
            this.gamma_constats[7] = this.gamma_constats[4] + Math.Log(this.gamma_constats[2] * this.gamma_constats[3] / this.gamma_constats[4]) - 2 * this.gamma_constats[0] - 3.7203285;
        }
        private double GO()
        {
            double result, U, E1, E2, S;
            int iter = 0;
            do
            {
                U = this.rand.NextDouble();
                if (U <= 0.0095722652)
                {
                    E1 = Exponential(); E2 = Exponential();
                    result = this.gamma_constats[4] * (1 + E1 / this.gamma_constats[3]);
                    if (this.gamma_constats[0] * (result / this.gamma_constats[4] - Math.Log(result / this.gamma_constats[0])) + this.gamma_constats[7] <= E2) return result;
                }
                else {
                    do {
                        E1 = Normal();
                        result = this.gamma_constats[2] * E1 + this.gamma_constats[0];
                    } while (result < 0 || result > this.gamma_constats[4]);
                    U = this.rand.NextDouble();
                    S = 0.5 * E1 * E1;
                    if (E1 > 0)
                        if (U < 1 - this.gamma_constats[5] * S) return result;
                        else if (U < 1 + this.gamma_constats[5] * (this.gamma_constats[6]) * E1 - this.gamma_constats[5]) return result;
                    if (Math.Log(U) < this.gamma_constats[0] * Math.Log(result / this.gamma_constats[0]) + this.gamma_constats[0] - result + S) return result;
                }
            } while (++iter < 1e9);
            throw new Exception("Failed to calculate GO");
        }
        public double Gamma(double loc = 0.0, double scale = 1.0, double k = 1.0) {
            double result;
            if (double.IsInteger(k) && k < 5)
                result = GA1((int)k);
            else if (double.IsInteger(2 * k) && k < 5)
                result = GA2(k);
            else if (k < 1)
                result = GS(k);
            else if (k > 1 && k < 3)
                result = GF(k);
            else
            {
                InitConstantGO(k);
                result = GO();
            }
            return scale * result + loc;
        }
        public Vectors Gamma((int, int) Shape, double loc = 0.0, double scale = 1.0, double k = 1.0) {
            Vectors result = Vectors.InitVectors(Shape);
            if (k > 3 && !(k < 5 && (double.IsInteger(k) || double.IsInteger(2 * k))))
            {
                InitConstantGO(k);
                for (int i = 0; i < Shape.Item1; i++)
                    for (int j = 0; j < Shape.Item2; j++)
                        result[i, j] = scale * GO() + loc;
            }
            else
            {
                for (int i = 0; i < Shape.Item1; i++)
                    for (int j = 0; j < Shape.Item2; j++)
                        result[i, j] = Gamma(loc, scale, k);
            }
            return result;
        }
    }
}