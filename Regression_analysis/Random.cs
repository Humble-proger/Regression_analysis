namespace Regression_analysis
{
    internal static class MathConstants
    {
        public const double Sqrt2 = 1.4142135623730951;
        public const double Sqrt3 = 1.7320508075688772;
    }

    public enum TypeDisribution {
        Uniform,
        Normal,
        Exponential,
        Laplace,
        Cauchy,
        Gamma
    }

    public interface IRandomDistribution
    {
        public abstract static double Generate(in Random rand);
        public abstract static double? Generate(Vectors paramsDist, in Random rand);
        public abstract static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand);
        public double Generate();
        public double? Generate(Vectors paramsDist);
        public Vectors? Generate((int, int) shape, Vectors paramsDist);
        public abstract static bool CheckParamsDist(Vectors paramsDist);
    }

    public class UniformDistribution(int? seed = null) : IRandomDistribution
    {
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);

        public static bool CheckParamsDist(Vectors paramsDist)
        {
            return paramsDist.Size == 2 && paramsDist[0] < paramsDist[1];
        }
        private static double Uniform(double a, double b, in Random rand) => a + rand.NextDouble() * (b - a);
        public static double Generate(in Random rand) => rand.NextDouble();
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Uniform(paramsDist[0], paramsDist[1], rand) : null;
        public double Generate() => _random.NextDouble();
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Uniform(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist)
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var vec = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    vec[i, j] = Uniform(paramsDist[0], paramsDist[1], _random);
            return vec;
        }

        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var vec = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    vec[i, j] = Uniform(paramsDist[0], paramsDist[1], rand);
            return vec;
        }
    }

    public class ExponentialDistribution(int? seed = null) : IRandomDistribution {
        
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Exponential(double loc, double scale, double u) => -Math.Log(1 - u) * scale + loc;
        public static bool CheckParamsDist(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public double Generate() => Exponential(0, 1, _random.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], _random.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Exponential(paramsDist[0], paramsDist[1], _random.NextDouble());
            return result;
        }
        public static double Generate(in Random rand) => Exponential(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Exponential(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Exponential(paramsDist[0], paramsDist[1], rand.NextDouble());
            return result;
        }
    }

    public class LaplaceDistribution(int? seed = null) : IRandomDistribution
    {
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        private static double Laplace(double loc, double scale, in Random rand) => loc - double.Sign(-1 + rand.NextDouble() * 2) * Math.Log(1 - rand.NextDouble()) * scale;
        public static bool CheckParamsDist(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public double Generate() => Laplace(0, 1, _random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Laplace(paramsDist[0], paramsDist[1], _random);
            return result;
        }
        public static double Generate(in Random rand) => Laplace(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Laplace(paramsDist[0], paramsDist[1], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Laplace(paramsDist[0], paramsDist[1], rand);
            return result;
        }
    }

    public class CauchyDistribution(int? seed = null) : IRandomDistribution
    {
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);

        private static double Cauchy(double loc, double scale, double u) => loc + scale * Math.Tan(double.Pi * (-1.5 + u * 2));
        public static bool CheckParamsDist(Vectors paramsDist) => paramsDist.Size == 2 && paramsDist[1] > 0;
        public double Generate() => Cauchy(0, 1, _random.NextDouble());
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], _random.NextDouble()) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Cauchy(paramsDist[0], paramsDist[1], _random.NextDouble());
            return result;
        }

        public static double Generate(in Random rand) => Cauchy(0, 1, rand.NextDouble());
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Cauchy(paramsDist[0], paramsDist[1], rand.NextDouble()) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Cauchy(paramsDist[0], paramsDist[1], rand.NextDouble());
            return result;
        }
    }

    public class NormalDistribution(int? seed = null) : IRandomDistribution
    { 
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);

        private static double Uniform(double a, double b, in Random rand) => a + (b - a) * rand.NextDouble();
        private static double Normal(double loc, double scale, in Random rand) {
            double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
            var s = e1 * e1 + e2 * e2;
            while (s > 1 || s < double.Epsilon)
            {
                e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                s = e1 * e1 + e2 * e2;
            }
            return e1 * Math.Sqrt(-2 * Math.Log(s) / s) * scale + loc;
        }
        public static bool CheckParamsDist(Vectors paramsDist) => paramsDist.Size == 2 & paramsDist[1] > 0;
        public double Generate() => Normal(0, 1, _random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Normal(paramsDist[0], paramsDist[1], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
            {
                for (var j = 0; j < count_2; j++)
                {
                    double e1 = Uniform(-1, 1, _random), e2 = Uniform(-1, 1, _random);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, _random); e2 = Uniform(-1, 1, _random);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[i, 2 * j] = e1 * val + paramsDist[0];
                    result[i, 2 * j + 1] = e2 * val + paramsDist[0];
                }
            }
            if (shape.Item2 % 2 == 1)
            {
                for (var i = 0; i < count_1; i++)
                {
                    double e1 = Uniform(-1, 1, _random), e2 = Uniform(-1, 1, _random);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, _random); e2 = Uniform(-1, 1, _random);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[2 * i, -1] = e1 * val + paramsDist[0];
                    result[2 * i + 1, -1] = e2 * val + paramsDist[0];
                }
                if (shape.Item1 % 2 == 1) result[-1, -1] = Normal(paramsDist[0], paramsDist[1], _random);
            }
            return result;
        }

        public static double Generate(in Random rand) => Normal(0, 1, rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Normal(paramsDist[0], paramsDist[1], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            int count_1 = shape.Item1 / 2, count_2 = shape.Item2 / 2;
            double val;
            for (var i = 0; i < shape.Item1; i++)
            {
                for (var j = 0; j < count_2; j++)
                {
                    double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[i, 2 * j] = e1 * val + paramsDist[0];
                    result[i, 2 * j + 1] = e2 * val + paramsDist[0];
                }
            }
            if (shape.Item2 % 2 == 1)
            {
                for (var i = 0; i < count_1; i++)
                {
                    double e1 = Uniform(-1, 1, rand), e2 = Uniform(-1, 1, rand);
                    var s = e1 * e1 + e2 * e2;
                    while (s > 1 || s < double.Epsilon)
                    {
                        e1 = Uniform(-1, 1, rand); e2 = Uniform(-1, 1, rand);
                        s = e1 * e1 + e2 * e2;
                    }
                    val = Math.Sqrt(-2 * Math.Log(s) / s) * paramsDist[1];
                    result[2 * i, -1] = e1 * val + paramsDist[0];
                    result[2 * i + 1, -1] = e2 * val + paramsDist[0];
                }
                if (shape.Item1 % 2 == 1) result[-1, -1] = Normal(paramsDist[0], paramsDist[1], rand);
            }
            return result;
        }
    }

    public class GammaDistribution(int? seed = null) : IRandomDistribution 
    {
        private readonly Random _random = seed is null ? new Random() : new Random((int) seed);
        
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Uniform(Vectors paramDist, in Random rand) => (double) UniformDistribution.Generate(paramDist, rand);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
        private static double Exponential(in Random rand) => ExponentialDistribution.Generate(rand);
        private static double Normal(in Random rand) => NormalDistribution.Generate(rand);

        private static double GA1(in Random rand, int k = 1)
        {
            var res = 0.0;
            for (var i = 0; i < k; i++)
                res += Exponential(rand);
            return res;
        }
        private static double GA2(in Random rand ,double k = 0.5)
        {
            var res = Normal(rand);
            var i = 1;
            while (i < k)
            {
                res += Exponential(rand); i++;
            }
            return res;
        }
        private static double GS(in Random rand, double k)
        {
            double res, u, w;
            var iter = 0;
            do
            {
                u = Uniform(new Vectors([0, 1 + k / double.E]), rand);
                w = Exponential(rand);
                if (u <= 1)
                {
                    res = Math.Pow(u, 1.0 / k);
                    if (res <= w)
                        return res;
                }
                else
                {
                    res = -Math.Log((1 - u) / k + 1.0 / double.E);
                    if ((1 - k) * Math.Log(res) <= w)
                        return res;
                }
            } while (++iter < 1e9);
            throw new Exception("Failed to calculate GS");
        }
        private static double GF(in Random rand, double k)
        {
            double e1, e2;
            do
            {
                e1 = Exponential(rand); e2 = Exponential(rand);
            } while (e2 < (k - 1) * (e1 - Math.Log(e1) - 1));
            return k * e1;
        }
        private static double[] InitConstantGO(double k)
        {
            var val = Math.Sqrt(k);
            var gamma_constats = new double[8];
            gamma_constats[0] = k - 1;
            gamma_constats[1] = 2 * MathConstants.Sqrt2 * val / MathConstants.Sqrt3 + k;
            gamma_constats[2] = Math.Sqrt(gamma_constats[1]);
            gamma_constats[3] = MathConstants.Sqrt2 * MathConstants.Sqrt3 * gamma_constats[1];
            gamma_constats[4] = gamma_constats[3] + gamma_constats[0];
            gamma_constats[5] = gamma_constats[1] / (gamma_constats[0] - 1);
            gamma_constats[6] = 2 * gamma_constats[1] / (gamma_constats[0] * val);
            gamma_constats[7] = gamma_constats[4] + Math.Log(gamma_constats[2] * gamma_constats[3] / gamma_constats[4]) - 2 * gamma_constats[0] - 3.7203285;
            return gamma_constats;
        }
        private static double GO(in double[] gamma_constats, in Random rand)
        {
            double result, u, e1, e2, s;
            var iter = 0;
            do
            {
                u = rand.NextDouble();
                if (u <= 0.0095722652)
                {
                    e1 = Exponential(rand); e2 = Exponential(rand);
                    result = gamma_constats[4] * (1 + e1 / gamma_constats[3]);
                    if (gamma_constats[0] * (result / gamma_constats[4] - Math.Log(result / gamma_constats[0])) + gamma_constats[7] <= e2) return result;
                }
                else
                {
                    do
                    {
                        e1 = Normal(rand);
                        result = gamma_constats[2] * e1 + gamma_constats[0];
                    } while (result < 0 || result > gamma_constats[4]);
                    u = rand.NextDouble();
                    s = 0.5 * e1 * e1;
                    if (e1 > 0)
                        if (u < 1 - gamma_constats[5] * s) return result;
                        else if (u < 1 + gamma_constats[5] * gamma_constats[6] * e1 - gamma_constats[5]) return result;
                    if (Math.Log(u) < gamma_constats[0] * Math.Log(result / gamma_constats[0]) + gamma_constats[0] - result + s) return result;
                }
            } while (++iter < 1e9);
            throw new Exception("Failed to calculate GO");
        }

        private static double Gamma(double loc, double scale, double k, in Random rand) {
            double result;
            if (double.IsInteger(k) && k < 5)
                result = GA1(rand, (int) k);
            else if (double.IsInteger(2 * k) && k < 5)
                result = GA2(rand, k);
            else if (k < 1)
                result = GS(rand, k);
            else if (k > 1 && k < 3)
                result = GF(rand, k);
            else
            {
                var constant = InitConstantGO(k);
                result = GO(constant, rand);
            }
            return scale * result + loc;
        }

        public static bool CheckParamsDist(Vectors paramsDist) => paramsDist.Size == 3 && paramsDist[1] > 0 && paramsDist[2] > 0;
        public double Generate() => GA1(_random);
        public double? Generate(Vectors paramsDist) => CheckParamsDist(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], _random) : null;
        public Vectors? Generate((int, int) shape, Vectors paramsDist) 
        {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], _random);
            return result;
        }
        public static double Generate(in Random rand) => GA1(rand);
        public static double? Generate(Vectors paramsDist, in Random rand) => CheckParamsDist(paramsDist) ? Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand) : null;
        public static Vectors? Generate((int, int) shape, Vectors paramsDist, in Random rand) {
            if (!CheckParamsDist(paramsDist)) return null;
            var result = Vectors.InitVectors(shape);
            for (var i = 0; i < shape.Item1; i++)
                for (var j = 0; j < shape.Item2; j++)
                    result[i, j] = Gamma(paramsDist[0], paramsDist[1], paramsDist[2], rand);
            return result;
        }
    }

    public static class LinespaceRandom
    {
        public static T[] Shuffle<T>(T[] list, in Random rand)
        {
            var n = list.Length;
            while (n > 1)
            {
                n--;
                var k = rand.Next(n + 1);
                (list[n], list[k]) = (list[k], list[n]);
            }
            return list;
        }

        public static Vectors Generate((int, int) shape, (double, double)[] intervals, in Random rand)
        {
            if (intervals.Length != shape.Item2)
                throw new ArgumentException("Количество элементов в Intervals должно совпадать с количеством столбцов");

            var step = new double[shape.Item2];
            for (var i = 0; i < shape.Item2; i++)
                step[i] = (intervals[i].Item2 - intervals[i].Item1) / (shape.Item1 - 1);

            var values = new int[shape.Item1];
            for (var i = 0; i < shape.Item1; i++)
                values[i] = i;

            var result = Vectors.InitVectors(shape);
            int[] copyValues;
            for (var j = 0; j < shape.Item2; j++)
            {
                copyValues = Shuffle((int[]) values.Clone(), rand);
                for (var i = 0; i < shape.Item1; i++)
                    result[i, j] = intervals[j].Item1 + copyValues[i] * step[j];
            }
            return result;
        }
    }
}