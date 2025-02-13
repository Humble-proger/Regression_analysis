using Regression_analysis;
using System.Diagnostics;

namespace Regression_analysis
{
    interface IOptimisation {
        public int MaxIter { get; set; }
        public Vectors Arg { get; set; }
        public double Tolerance { get; set; }
        public OptRes Optimisate(Vectors InitParam);
    }

    struct OptRes {
        public Vectors MinPoint;
        public double MinValueFunction;
        public int NumIteration;
        public double tol;
        public int CountCalcFunc;
        public bool Convergence;
        public double? Norm;
        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nЗначение функции: {MinValueFunction};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {tol}";
        }
    }
    struct OptResExtended
    {
        public Vectors MinPoint;
        public double MinValueFunction;
        public int NumIteration;
        public double tol;
        public int CountCalcFunc;
        public bool Convergence;
        public double? Norm;
        public Vectors InitParametrs;
        public int NumberRebounds;

        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nЗначение функции: {MinValueFunction};\nНачальные значения: {InitParametrs};\nКоличество подбора начальных значений: {NumberRebounds};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {tol}";
        }
    }

    class QuadraticInterpolation(Func<Vectors, Vectors, double> func, Vectors xk, Vectors pk, Vectors arg, int maxiter = 100, double tol = 1e-7) : IOptimisation
    {
        private readonly Func<Vectors, Vectors, double> FuncToOptimisate = func;
        private Vectors _args = arg;
        public Vectors Xk { get; set; } = xk;
        public Vectors Pk { get; set; } = pk;

        public int MaxIter { get; set; } = maxiter;
        public Vectors Arg { get => _args; set => _args = value; }
        public double Tolerance { get; set; } = tol;

        private static double Min(params double[] numbers) => numbers.Min();
        public OptRes Optimisate(Vectors InitParam)
        {
            OptRes result = new()
            {
                tol = Tolerance,
                CountCalcFunc = 0,
                NumIteration = 0,
                Convergence = false
            };
            double Func(double alpha) => FuncToOptimisate(Xk + alpha * Pk, _args);
            if (maxiter < 1) {
                result.MinPoint = new Vectors([(InitParam[0] + InitParam[1]) / 2]);
                result.MinValueFunction = Func(result.MinPoint[0]); 
                return result; 
            };
            double[] X = [InitParam[0], (InitParam[0] + InitParam[1]) / 2, InitParam[1]];
            double[] F = [Func(X[0]), Func(X[1]), Func(X[2])];
            (double, double)[] temp_arr = [(0, 0.5)];
            result.CountCalcFunc += 3;
            double Numerator, Denominator, Xmin, temp1, temp2, temp3, temp4, FuncMin;
            for (; result.NumIteration < maxiter; result.NumIteration++)
            {
                temp1 = X[1] - X[0]; temp2 = X[1] - X[2];
                temp3 = F[1] - F[0]; temp4 = F[1] - F[2];
                Numerator = temp1 * temp1 * temp4 - temp2 * temp2 * temp3;
                Denominator = 2 * (temp1 * temp4 - temp2 * temp3);
                if (double.Abs(Denominator) < double.Epsilon) {
                    result.MinPoint = new Vectors([temp_arr[0].Item2]);
                    result.MinValueFunction = temp_arr[0].Item1;
                    return result;
                }
                Xmin = X[1] - Numerator / Denominator;
                FuncMin = Func(Xmin); result.CountCalcFunc++;
                temp_arr = [(F[0], X[0]), (F[1], X[1]), (F[2], X[2]), (FuncMin, Xmin)];
                temp_arr = [.. temp_arr.OrderBy(x => x.Item1)];
                if (double.Abs((Min(F[0], F[1], F[2]) - FuncMin) / FuncMin) < tol)
                {
                    result.MinPoint = new Vectors([temp_arr[0].Item2]);
                    result.MinValueFunction = temp_arr[0].Item1;
                    result.Convergence = true;
                    return result;

                }
                X = (from v in temp_arr select v.Item2).Take(3).ToArray();
                F = (from v in temp_arr select v.Item1).Take(3).ToArray();
            }
            result.MinPoint = new Vectors([temp_arr[0].Item2]);
            result.MinValueFunction = temp_arr[0].Item1;
            return result;
        }
    }


    class CG_method : IOptimisation
    {
        private readonly Func<Vectors, Vectors, double> FuncToOptimisate;
        private readonly Func<Vectors, Vectors, Vectors> dfFuncToOptimisate;
        private readonly QuadraticInterpolation FindAlpha;
        private int _maxiter = 100;
        private Vectors _args;
        private double _tolerance = 1e-7;
        private readonly double _maxalpha = 1.0;
        private readonly double _minalpha = 1e-8;

        public int MaxIter { get => _maxiter; set => _maxiter = value; }
        public Vectors Arg { get => _args; set => _args = value; }
        public double Tolerance { get => _tolerance; set => _tolerance = value; }

        public CG_method(Func<Vectors, Vectors, double> func, Func<Vectors, Vectors, Vectors> df_func, Vectors? args = null, int maxiter = 100, double tol = 1e-7, double maxalpha = 1, double minalpha = double.Epsilon)
        {
            FuncToOptimisate = func;
            dfFuncToOptimisate = df_func;
            _maxiter = maxiter;
            if (args != null)
                _args = args;
            else _args = Vectors.InitVectors((1, 1));
            _tolerance = tol;
            _maxalpha = maxalpha;
            _minalpha = minalpha;
            FindAlpha = new QuadraticInterpolation(FuncToOptimisate, Vectors.Zeros((1, 1)), Vectors.Zeros((1, 1)), _args, tol: 1e-3);
        }
        private static double ScalarMult(Vectors v1, Vectors v2) {
            if (v1.Shape.Item1 == 1 && v2.Shape.Item1 == 1 && v1.Shape.Item2 == v2.Shape.Item2 ) {
                double summator = 0.0;
                for (int i = 0; i < v1.Shape.Item2; i++)
                    summator += v1[0, i] * v2[0, i];
                return summator;
            }
            else throw new Exception("Incorrect shapes vectors.");
        }
      
        public OptRes Optimisate(Vectors InitParam)
        {
            double AlphaK = 1.0, w;
            Vectors xk = InitParam, xkp1, gfk = dfFuncToOptimisate(xk, _args), gfkp1, pk = -gfk, MinMaxAlpha = new([_minalpha, _maxalpha]);
            int iter = 0, CalcFunc = 0; double _norm;
            OptRes LinRes;
            for (; (_norm = Vectors.Norm(pk)) > _tolerance && iter < _maxiter; iter++) {
                FindAlpha.Xk = xk; FindAlpha.Pk = pk;
                LinRes = FindAlpha.Optimisate(MinMaxAlpha);
                AlphaK = LinRes.MinPoint[0];
                CalcFunc += LinRes.CountCalcFunc;
                xkp1 = xk + AlphaK * pk;
                gfkp1 = dfFuncToOptimisate(xkp1, _args);
                w = ScalarMult(gfkp1, gfkp1) / ScalarMult(gfk, gfk);
                pk = (gfkp1 - w * pk) * -1;
                (xk, gfk) = (xkp1, gfkp1);
            }
            OptRes temp_res = new()
            {
                MinPoint = xk,
                MinValueFunction = FuncToOptimisate(xk, _args),
                CountCalcFunc = CalcFunc,
                NumIteration = iter,
                tol = _tolerance,
                Norm = _norm
            };
            if (Vectors.Norm(pk) < _tolerance)
                temp_res.Convergence = true;
            else
                temp_res.Convergence = false;
            return temp_res;
        }

        public OptResExtended OptimisateRandomInit((int, int) ShapeInitParam, double MinValue = 0, double MaxValue = 100,int MaxIter = 100, int? seed = null) {
            Random_distribution rand;
            if (seed == null)
                rand = new Random_distribution();
            else
                rand = new Random_distribution(seed);
            OptRes resMethod;
            OptResExtended Result = new()
            {
                tol = _tolerance,
                CountCalcFunc = 0,
                Convergence = false,
                NumIteration = 0,
                NumberRebounds = 0,
                Norm = double.MaxValue
            };
            for (; Result.NumberRebounds < MaxIter && !Result.Convergence; Result.NumberRebounds++) {
                Result.InitParametrs = rand.Uniform(MinValue, MaxValue, ShapeInitParam);
                resMethod = Optimisate(Result.InitParametrs);
                if (resMethod.Convergence)
                {
                    Result.Convergence = true;
                    Result.MinPoint = resMethod.MinPoint;
                    Result.MinValueFunction = resMethod.MinValueFunction;
                    Result.NumIteration = resMethod.NumIteration;
                    Result.Norm = resMethod.Norm;
                    Result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                else {
                    if (resMethod.Norm < Result.Norm) {
                        Result.Norm = resMethod.Norm;
                        Result.MinPoint = resMethod.MinPoint;
                        Result.MinValueFunction = resMethod.MinValueFunction;
                        Result.NumIteration = resMethod.NumIteration;
                    }
                    Result.CountCalcFunc += resMethod.CountCalcFunc;
                }
            }
            return Result;
        }
    }
}

class DFP_method : IOptimisation
{
    private readonly Func<Vectors, Vectors, double> FuncToOptimisate;
    private readonly Func<Vectors, Vectors, Vectors> dfFuncToOptimisate;
    private readonly QuadraticInterpolation FindAlpha;
    private int _maxiter = 100;
    private Vectors _args;
    private double _tolerance = 1e-7;
    private readonly double _maxalpha = 1.0;
    private readonly double _minalpha = 1e-8;

    public int MaxIter { get => _maxiter; set => _maxiter = value; }
    public Vectors Arg { get => _args; set => _args = value; }
    public double Tolerance { get => _tolerance; set => _tolerance = value; }

    public DFP_method(Func<Vectors, Vectors, double> func, Func<Vectors, Vectors, Vectors> df_func, Vectors? args = null, int maxiter = 100, double tol = 1e-7, double maxalpha = 100, double minalpha = 1e-7)
    {
        FuncToOptimisate = func;
        dfFuncToOptimisate = df_func;
        _maxiter = maxiter;
        if (args != null)
            _args = args;
        else _args = Vectors.InitVectors((1, 1));
        _tolerance = tol;
        _maxalpha = maxalpha;
        _minalpha = minalpha;
        FindAlpha = new QuadraticInterpolation(FuncToOptimisate, Vectors.Zeros((1, 1)), Vectors.Zeros((1, 1)), _args, tol: 1e-7);
    }
    private static double ScalarMult(Vectors v1, Vectors v2)
    {
        if (v1.Shape.Item1 == 1 && v2.Shape.Item1 == 1 && v1.Shape.Item2 == v2.Shape.Item2)
        {
            double summator = 0.0;
            for (int i = 0; i < v1.Shape.Item2; i++)
                summator += v1[0, i] * v2[0, i];
            return summator;
        }
        else throw new Exception("Incorrect shapes vectors.");
    }

    public OptRes Optimisate(Vectors InitParam)
    {
        if (!InitParam.IsVector()) throw new Exception("InitParam должен быть вектором.");
        Vectors Hk = Vectors.Eig((InitParam.Size, InitParam.Size)), Xk = InitParam, Xkp1, gfk = dfFuncToOptimisate(Xk, Arg), gfkp1, 
            pk, MinMaxAlpha = new([_minalpha, _maxalpha]), sk, rk, A, B;
        double alpha_k;
        OptRes optRes;
        OptRes Result = new () {
            tol = _tolerance,
            CountCalcFunc = 0,
            Convergence = false,
            NumIteration = 0,
            Norm = double.MaxValue
        };
        //Console.WriteLine("Начало оптимизации...");
        for (; Result.NumIteration < MaxIter && (Result.Norm = Vectors.Norm(gfk)) > Tolerance; Result.NumIteration++) {
            //Console.WriteLine($"---------- Итерация {Result.NumIteration} ----------");
            pk = -Hk.Dot(gfk.T()).T();
            //Console.WriteLine($"pk: {pk}");
            FindAlpha.Xk = Xk; FindAlpha.Pk = pk;
            optRes = FindAlpha.Optimisate(MinMaxAlpha);
            //Console.WriteLine(optRes);
            Result.CountCalcFunc += optRes.CountCalcFunc;
            alpha_k = optRes.MinPoint[0];
            Xkp1 = Xk + alpha_k * pk;
            gfkp1 = dfFuncToOptimisate(Xkp1, Arg);
            sk = alpha_k * pk;
            rk = gfkp1 - gfk;
            //Console.WriteLine($"sk: {sk}; rk: {rk}");

            // Считаем Гессиан
            A = (sk.T() & sk) / ScalarMult(sk, rk);
            B = (Hk & (rk.T() & rk) & Hk) / ScalarMult(rk & Hk, rk);
            //Console.WriteLine($"A:\n{A}; B:\n{B}");
            Hk = Hk + A - B;
            //Console.WriteLine($"Hk:\n{Hk}");
            Xk = Xkp1; gfk = gfkp1;
            //Console.WriteLine("---------------------");
        }
        if (Result.Norm < Tolerance) {
            Result.Convergence = true;
        }
        Result.MinPoint = Xk;
        Result.MinValueFunction = FuncToOptimisate(Xk, Arg);
        return Result;
    }
    public OptResExtended OptimisateRandomInit((int, int) ShapeInitParam, double MinValue = -10, double MaxValue = 10, int MaxIter = 100, int? seed = null)
    {
        Random_distribution rand;
        if (seed == null)
            rand = new ();
        else
            rand = new (seed);
        OptRes resMethod;
        OptResExtended Result = new()
        {
            tol = _tolerance,
            CountCalcFunc = 0,
            Convergence = false,
            NumIteration = 0,
            NumberRebounds = 0,
            Norm = double.MaxValue
        };
        for (; Result.NumberRebounds < MaxIter && !Result.Convergence; Result.NumberRebounds++)
        {
            Result.InitParametrs = rand.Uniform(MinValue, MaxValue, ShapeInitParam);
            resMethod = Optimisate(Result.InitParametrs);
            if (resMethod.Convergence)
            {
                Result.Convergence = true;
                Result.MinPoint = resMethod.MinPoint;
                Result.MinValueFunction = resMethod.MinValueFunction;
                Result.NumIteration = resMethod.NumIteration;
                Result.Norm = resMethod.Norm;
                Result.CountCalcFunc += resMethod.CountCalcFunc;
            }
            else
            {
                if (resMethod.Norm < Result.Norm)
                {
                    Result.Norm = resMethod.Norm;
                    Result.MinPoint = resMethod.MinPoint;
                    Result.MinValueFunction = resMethod.MinValueFunction;
                    Result.NumIteration = resMethod.NumIteration;
                }
                Result.CountCalcFunc += resMethod.CountCalcFunc;
            }
        }
        return Result;
    }
}

abstract class TestFunction {
    public static double Function1(Vectors vec, Vectors _) {
        double tmp = vec[1] - vec[0], tmp2 = 1 - vec[0];
        return 100 * tmp * tmp + tmp2 * tmp2;
    }
    public static double Function2(Vectors vec, Vectors _) {
        double tmp = vec[1] - vec[0] * vec[0], tmp2 = 1 - vec[0];
        return 100 * tmp * tmp + tmp2 * tmp2;
    }
    public static double Function3(Vectors vec, Vectors _)
    {
        double tmp = (vec[1] - 2) / 3, tmp1 = (vec[1] - 2) / 3, tmp2 = vec[0] - 1, tmp3 = (vec[1] - 1) / 2;
        return -(1 / (1 + tmp * tmp + tmp1 * tmp1) + 3 / (1 + tmp2 * tmp2 + tmp3 * tmp3));
    }
    public static Vectors DFunction1(Vectors vec, Vectors _)
    {
        return new Vectors([-200 * vec[1] + 202 * vec[0] - 2, 200 * (vec[1] - vec[0])]);
    }
    public static Vectors DFunction2(Vectors vec, Vectors _) {
        double tmp = vec[1] - vec[0] * vec[0];
        return new Vectors([-400 * tmp * vec[0] + 2 * vec[0] - 2, 200 * tmp]);
    }
    public static Vectors DFunction3(Vectors vec, Vectors _)
    {
        double tmp = (vec[1] - 2) / 3, tmp1 = (vec[1] - 2) / 3, tmp2 = vec[0] - 1, tmp3 = (vec[1] - 1) / 2;
        double denominator1 = 1 + tmp * tmp + tmp1 * tmp1, denominator2 = 1 + tmp2 * tmp2 + tmp3 * tmp3;
        denominator1 *= denominator1; denominator2 *= denominator2;
        return (new Vectors([-(2*vec[0] - 4)/(9 * denominator1) - (6*(vec[0] - 1))/denominator2,
                        -(2*vec[1] - 4)/(9 * denominator1) - (3*(vec[1] - 1)) / (2*denominator2)])) * -1;
    }
}

class TestClass {
    static void Main() {
        var opt1 = new DFP_method(TestFunction.Function1, TestFunction.DFunction1);
        var opt2 = new CG_method(TestFunction.Function1, TestFunction.DFunction1);
        var clock = new Stopwatch();
        var rand = new Random();
        var seed = rand.Next();
        clock.Start();
        var Res = opt1.OptimisateRandomInit((1, 2), seed : seed);
        clock.Stop();
        Console.WriteLine(Res);
        Console.WriteLine($"Время работы: {clock.ElapsedTicks}");
        clock.Restart();
        Res = opt2.OptimisateRandomInit((1, 2), seed: seed);
        clock.Stop();
        Console.WriteLine(Res);
        Console.WriteLine($"Время работы: {clock.ElapsedTicks}");
    }
}