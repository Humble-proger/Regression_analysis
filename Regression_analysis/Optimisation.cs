
namespace Regression_analysis
{

    public interface IOprimizator {
        string Name { get; }

        bool IsUseGradient { get; }

        public abstract OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors x0,
            Vectors[] @params,
            double eps,
            int maxIter = 500
            );
        public abstract OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            (int, int) shapeInitParam,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            );
    }

    public struct OptRes
    {
        public Vectors MinPoint;
        public double MinValueFunction;
        public int NumIteration;
        public double Tol;
        public int CountCalcFunc;
        public bool Convergence;
        public double? Norm;
        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nЗначение функции: {MinValueFunction};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {Tol}";
        }
    }
    public struct OptResExtended
    {
        public Vectors MinPoint;
        public double MinValueFunction;
        public int NumIteration;
        public double Tol;
        public int CountCalcFunc;
        public bool Convergence;
        public double? Norm;
        public Vectors InitParametrs;
        public int NumberRebounds;

        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nЗначение функции: {MinValueFunction};\nНачальные значения: {InitParametrs};\nКоличество подбора начальных значений: {NumberRebounds};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {Tol}";
        }
    }

    public static class QuadraticInterpolation
    {
        private static double Min(params double[] numbers) => numbers.Min();
        public static OptRes Optimisate(
            LogLikelihoodFunction func,
            IModel model,
            Vectors[] @params,
            Vectors xk,
            Vectors pk,
            double eps= 1e-7,
            int maxIter = 100,
            double minalpha = 1e-8,
            double maxalpha = 1.0)
        {
            OptRes result = new()
            {
                Tol = eps,
                CountCalcFunc = 0,
                NumIteration = 0,
                Convergence = false
            };
            double Func(double alpha) => func(model, xk + alpha * pk, @params);
            if (maxIter < 1)
            {
                result.MinPoint = new Vectors([(minalpha + maxalpha) / 2]);
                result.MinValueFunction = Func(result.MinPoint[0]);
                return result;
            };
            double[] x = [minalpha, (minalpha + maxalpha) / 2, maxalpha];
            double[] f = [Func(x[0]), Func(x[1]), Func(x[2])];
            (double, double)[] temp_arr = [(0, 0.5)];
            result.CountCalcFunc += 3;
            double numerator, denominator, xmin, temp1, temp2, temp3, temp4, funcMin;
            for (; result.NumIteration < maxIter; result.NumIteration++)
            {
                temp1 = x[1] - x[0]; temp2 = x[1] - x[2];
                temp3 = f[1] - f[0]; temp4 = f[1] - f[2];
                numerator = temp1 * temp1 * temp4 - temp2 * temp2 * temp3;
                denominator = 2 * (temp1 * temp4 - temp2 * temp3);
                if (double.Abs(denominator) < double.Epsilon)
                {
                    result.MinPoint = new Vectors([temp_arr[0].Item2]);
                    result.MinValueFunction = temp_arr[0].Item1;
                    return result;
                }
                xmin = x[1] - numerator / denominator;
                funcMin = Func(xmin); result.CountCalcFunc++;
                temp_arr = [(f[0], x[0]), (f[1], x[1]), (f[2], x[2]), (funcMin, xmin)];
                temp_arr = [.. temp_arr.OrderBy(x => x.Item1)];
                if (double.Abs((Min(f[0], f[1], f[2]) - funcMin) / funcMin) < eps)
                {
                    result.MinPoint = new Vectors([temp_arr[0].Item2]);
                    result.MinValueFunction = temp_arr[0].Item1;
                    result.Convergence = true;
                    return result;

                }
                x = (from v in temp_arr select v.Item2).Take(3).ToArray();
                f = (from v in temp_arr select v.Item1).Take(3).ToArray();
            }
            result.MinPoint = new Vectors([temp_arr[0].Item2]);
            result.MinValueFunction = temp_arr[0].Item1;
            return result;
        }
    }


    public class CGOptimizator : IOprimizator
    {

        public string Name => "CG";

        public bool IsUseGradient => true;

        private static double ScalarMult(Vectors v1, Vectors v2)
        {
            if (v1.Shape.Item1 == 1 && v2.Shape.Item1 == 1 && v1.Shape.Item2 == v2.Shape.Item2)
            {
                var summator = 0.0;
                for (var i = 0; i < v1.Shape.Item2; i++)
                    summator += v1[0, i] * v2[0, i];
                return summator;
            }
            else throw new Exception("Incorrect shapes vectors.");
        }

        public OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors x0,
            Vectors[] @params,
            double eps,
            int maxIter = 500
            )
        {
            if (dffunc is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");
            double alphaK = 1.0, w;
            Vectors xk = x0, xkp1, gfk = dffunc(model, x0, @params), gfkp1, pk = -gfk;
            int iter = 0, calcFunc = 0; double norm = double.MaxValue;
            OptRes linRes;

            Vectors DfFunc(Vectors x0) => dffunc(model, x0, @params);

            OptRes FindOptAlpha(Vectors xk, Vectors pk) => QuadraticInterpolation.Optimisate(func, model, @params, xk, pk);

            for (; norm > eps && iter < maxIter; iter++)
            {
                linRes = FindOptAlpha(xk, pk);
                alphaK = linRes.MinPoint[0];
                calcFunc += linRes.CountCalcFunc;
                xkp1 = xk + alphaK * pk;
                gfkp1 = DfFunc(xkp1);
                w = ScalarMult(gfkp1, gfkp1) / ScalarMult(gfk, gfk);
                pk = (gfkp1 - w * pk) * -1;
                norm = Vectors.Norm(xkp1 - xk);
                (xk, gfk) = (xkp1, gfkp1);
            }
            OptRes temp_res = new()
            {
                MinPoint = xk,
                MinValueFunction = func(model, xk, @params),
                CountCalcFunc = calcFunc,
                NumIteration = iter,
                Tol = eps,
                Norm = norm,
                Convergence = Vectors.Norm(pk) < eps
            };
            return temp_res;
        }

        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            (int, int) shapeInitParam,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            )
        {
            if (dffunc is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");
            var rand = seed == null ? new UniformDistribution() : new UniformDistribution(seed);
            OptRes resMethod;
            OptResExtended result = new()
            {
                Tol = eps,
                CountCalcFunc = 0,
                Convergence = false,
                NumIteration = 0,
                NumberRebounds = 0,
                Norm = double.MaxValue
            };
            var tmp = new Vectors([minValue, maxValue]);
            for (; result.NumberRebounds < maxIter && !result.Convergence; result.NumberRebounds++)
            {
                #pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                result.InitParametrs = rand.Generate(shapeInitParam, tmp);
                #pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                #pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
                #pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
                    result.MinValueFunction = resMethod.MinValueFunction;
                    result.NumIteration = resMethod.NumIteration;
                    result.Norm = resMethod.Norm;
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                else
                {
                    if (resMethod.Norm < result.Norm)
                    {
                        result.Norm = resMethod.Norm;
                        result.MinPoint = resMethod.MinPoint;
                        result.MinValueFunction = resMethod.MinValueFunction;
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                Console.WriteLine(result.NumberRebounds);
                Console.WriteLine(resMethod.Norm);
                Console.WriteLine(resMethod.MinPoint);
            }
            return result;
        }
    }

    public class DFPOptimizator : IOprimizator
    {
        public string Name => "DFP";

        public bool IsUseGradient => true;

        private static double ScalarMult(Vectors v1, Vectors v2)
        {
            if (v1.Shape.Item1 == 1 && v2.Shape.Item1 == 1 && v1.Shape.Item2 == v2.Shape.Item2)
            {
                var summator = 0.0;
                for (var i = 0; i < v1.Shape.Item2; i++)
                    summator += v1[0, i] * v2[0, i];
                return summator;
            }
            else throw new Exception("Incorrect shapes vectors.");
        }

        public OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors x0,
            Vectors[] @params,
            double eps,
            int maxIter = 500
            )
        {
            if (!x0.IsVector()) throw new Exception("InitParam должен быть вектором.");
            if (dffunc is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");
            Vectors DFunc(Vectors xk) => dffunc(model, xk, @params);

            Vectors hk = Vectors.Eig((x0.Size, x0.Size)), xk = x0, xkp1, gfk = DFunc(xk), gfkp1,
                pk, sk, rk, a, b;
            double alpha_k;
            OptRes optRes;
            OptRes result = new()
            {
                Tol = eps,
                CountCalcFunc = 0,
                Convergence = false,
                NumIteration = 0,
                Norm = double.MaxValue
            };

            double denominatorA, denominatorB;

            OptRes FindOptAlpha(Vectors xk, Vectors pk) => QuadraticInterpolation.Optimisate(func, model, @params, xk, pk);

            //Console.WriteLine("Начало оптимизации...");
            for (; result.NumIteration < maxIter && (result.Norm = Vectors.Norm(gfk)) > eps; result.NumIteration++)
            {
                //Console.WriteLine($"---------- Итерация {Result.NumIteration} ----------");
                pk = -hk.Dot(gfk.T()).T();
                //Console.WriteLine($"pk: {pk}");
                optRes = FindOptAlpha(xk, pk);
                //Console.WriteLine(optRes);
                result.CountCalcFunc += optRes.CountCalcFunc;
                alpha_k = optRes.MinPoint[0];
                xkp1 = xk + alpha_k * pk;
                gfkp1 = DFunc(xkp1);
                sk = alpha_k * pk;
                rk = gfkp1 - gfk;
                //Console.WriteLine($"sk: {sk}; rk: {rk}");

                // Считаем Гессиан
                denominatorA = ScalarMult(sk, rk);
                denominatorB = ScalarMult(rk & hk, rk);

                if (double.Abs(denominatorA) < eps || double.Abs(denominatorB) < eps)
                {
                    hk = Vectors.Eig(hk.Shape);
                }
                else 
                {
                    a = (sk.T() & sk) / denominatorA;
                    b = (hk & (rk.T() & rk) & hk) / denominatorB;
                    hk = hk + a - b;
                }
                xk = xkp1; gfk = gfkp1;
            }
            if (result.Norm < eps)
                result.Convergence = true;
            result.MinPoint = xk;
            result.MinValueFunction = func(model, xk, @params);
            return result;
        }
        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            (int, int) shapeInitParam,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            )
        {
            if (dffunc is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");
            var rand = seed == null ?  new() : (UniformDistribution) new(seed);
            OptRes resMethod;
            OptResExtended result = new()
            {
                Tol = eps,
                CountCalcFunc = 0,
                Convergence = false,
                NumIteration = 0,
                NumberRebounds = 0,
                Norm = double.MaxValue
            };
            var tmp = new Vectors([minValue, maxValue]);
            for (; result.NumberRebounds < maxIter && !result.Convergence; result.NumberRebounds++)
            {
                
                #pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                result.InitParametrs = rand.Generate(shapeInitParam, tmp);
                #pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                #pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
                #pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
                    result.MinValueFunction = resMethod.MinValueFunction;
                    result.NumIteration = resMethod.NumIteration;
                    result.Norm = resMethod.Norm;
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                else
                {
                    if (resMethod.Norm < result.Norm)
                    {
                        result.Norm = resMethod.Norm;
                        result.MinPoint = resMethod.MinPoint;
                        result.MinValueFunction = resMethod.MinValueFunction;
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                Console.WriteLine(result.NumberRebounds);
                Console.WriteLine(resMethod.Norm);
            }
            return result;
        }
    }

    public class NelderMeadOptimizator : IOprimizator
    {
        public string Name => "NelderMead";

        public bool IsUseGradient => false;
        public double Alpha { get; set; } = 1.0;
        public double Beta { get; set; } = 0.5;
        public double Gamma { get; set; } = 2.0;
        public double T { get; set; } = 1.0;

        public OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors x0,
            Vectors[] @params,
            double eps,
            int maxIter = 500
            )
        {
            OptRes result = new()
            {
                Tol = eps,
                CountCalcFunc = 0,
                Convergence = false,
                NumIteration = 0,
                Norm = double.MaxValue
            };
            if (!x0.IsVector() || x0.Shape.Item1 != 1)
                throw new ArgumentException("Initial point must be a row vector");

            int n = x0.Shape.Item2;
            if (n < 2)
                throw new ArgumentException("Dimension must be at least 2");

            // Инициализация симплекса (каждый СТОЛБЕЦ D — вершина размерности N)
            Vectors d = InitializeSimplex(x0, n, T);
            Vectors resF = EvaluateSimplex(func, ref model, ref @params, d, ref result.CountCalcFunc);

            while (result.NumIteration < maxIter)
            {
                int indexMin = Vectors.MinIndex(resF);
                int indexMax = Vectors.MaxIndex(resF);
                int indexSecondMax = Vectors.SecondMaxIndex(resF, indexMax);

                // Критерий останова
                if ((result.Norm = Vectors.NormDifference(resF, resF[indexMin])) < eps)
                    break;

                // Центр тяжести (исключая худшую вершину)
                Vectors xc = CalculateCentroid(d, n, indexMax);

                // Отражение
                Vectors xr = ReflectPoint(xc, Vectors.GetColumn(d, indexMax), Alpha);
                double fr = func(model, xr, @params);
                result.CountCalcFunc++;

                if (fr < resF[indexMin])
                {
                    // Растяжение
                    Vectors xe = ExpandPoint(xc, xr, Gamma);
                    double fe = func(model, xe, @params);
                    result.CountCalcFunc++;
                    UpdatePoint(d, resF, indexMax, (fe < fr) ? xe : xr, Math.Min(fe, fr));
                }
                else if (fr < resF[indexSecondMax])
                {
                    UpdatePoint(d, resF, indexMax, xr, fr);
                }
                else
                {
                    // Сжатие
                    Vectors xs = ContractPoint(xc, Vectors.GetColumn(d, indexMax), Beta);
                    double fs = func(model, xs, @params);
                    result.CountCalcFunc++;
                    if (fs < resF[indexMax])
                        UpdatePoint(d, resF, indexMax, xs, fs);
                    else
                        ShrinkSimplex(d, ref model, ref @params, resF, Vectors.GetColumn(d, indexMin), func, ref result.CountCalcFunc);
                }
                result.NumIteration++;
            }
            if (result.Norm < result.Tol)
                result.Convergence = true;
            result.MinPoint = Vectors.GetColumn(d, Vectors.MinIndex(resF));
            result.MinValueFunction = func(model, result.MinPoint, @params);
            return result;
        }

        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            (int, int) shapeInitParam,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            )
        {
            var rand = seed == null ?  new() : (UniformDistribution) new(seed);
            OptRes resMethod;
            var tmp = new Vectors([minValue, maxValue]);
            OptResExtended result = new()
            {
                Tol = eps,
                InitParametrs = x0 ?? Vectors.Zeros(shapeInitParam),
            };
            resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
            result.Convergence = resMethod.Convergence;
            result.NumIteration = resMethod.NumIteration;
            result.NumberRebounds += 1;
            result.Norm = resMethod.Norm;
            result.MinValueFunction = resMethod.MinValueFunction;
            result.MinPoint = resMethod.MinPoint;
            result.CountCalcFunc = resMethod.CountCalcFunc;
            for (; result.NumberRebounds < maxIter && !result.Convergence; result.NumberRebounds++)
            {
                #pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                result.InitParametrs = rand.Generate(shapeInitParam, tmp);
                #pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                #pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
                #pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
                    result.MinValueFunction = resMethod.MinValueFunction;
                    result.NumIteration = resMethod.NumIteration;
                    result.Norm = resMethod.Norm;
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
                else
                {
                    if (resMethod.Norm < result.Norm)
                    {
                        result.Norm = resMethod.Norm;
                        result.MinPoint = resMethod.MinPoint;
                        result.MinValueFunction = resMethod.MinValueFunction;
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
            }
            return result;
        }

        public OptResExtended OptimisateBestRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            (int, int) shapeInitParam,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 10,
            int? seed = null
            )
        {
            var rand = seed == null ? new() : (UniformDistribution) new(seed);
            OptRes resMethod;
            var tmp = new Vectors([minValue, maxValue]);
            var mnkestiminator = new MNKEstimator();
            OptResExtended result = new()
            {
                Tol = eps,
                InitParametrs = mnkestiminator.EstimateParameters(model, [@params[0], @params[1], @params[2]])
            };
            Console.WriteLine(result.InitParametrs);
            resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
            result.Convergence = resMethod.Convergence;
            result.NumIteration = resMethod.NumIteration;
            result.NumberRebounds += 1;
            result.Norm = resMethod.Norm;
            result.MinValueFunction = resMethod.MinValueFunction;
            result.MinPoint = resMethod.MinPoint;
            result.CountCalcFunc = resMethod.CountCalcFunc;
            Console.WriteLine(resMethod.MinPoint);
            double maxvalue = resMethod.MinValueFunction, minvalue = resMethod.MinValueFunction;
            Vectors initParam, minInitParam = result.InitParametrs;
            for (; result.NumberRebounds < maxIter; result.NumberRebounds++)
            {
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                initParam = rand.Generate(shapeInitParam, tmp);
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, initParam, @params, eps);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.MinValueFunction < result.MinValueFunction) {
                    result.Convergence = resMethod.Convergence;
                    result.MinPoint = resMethod.MinPoint;
                    result.MinValueFunction = resMethod.MinValueFunction;
                    result.NumIteration = resMethod.NumIteration;
                    result.Norm = resMethod.Norm;
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                    minValue = resMethod.MinValueFunction;
                    minInitParam = initParam;
                }
            }
            Console.WriteLine($"Min: {minValue} Max: {maxvalue} MinInitParam: {minInitParam}");
            return result;
        }

        private static Vectors InitializeSimplex(Vectors x0, int n, double t)
        {
            Vectors d = Vectors.InitVectors((n, n + 1));
            double d1 = t * (Math.Sqrt(n + 1) + n - 1) / (n * Math.Sqrt(2));
            double d2 = t * (Math.Sqrt(n + 1) - 1) / (n * Math.Sqrt(2));

            // Первая вершина — x0
            for (int i = 0; i < n; i++)
                d[i, 0] = x0[i];

            // Остальные вершины
            for (int i = 0; i < n; i++)
            {
                for (int j = 1; j <= n; j++)
                {
                    d[i, j] = i == j - 1 ? x0[i] + d1 : x0[i] + d2;
                }
            }
            return d;
        }

        private static Vectors EvaluateSimplex(LogLikelihoodFunction func, ref IModel model, ref Vectors[] @params, Vectors d, ref int countFuncEvals)
        {
            int verticesCount = d.Shape.Item2; // Число столбцов (N+1 вершин)

            Vectors resF = Vectors.Zeros((1, verticesCount)); // Вектор значений функции

            for (int i = 0; i < verticesCount; i++)
            {
                Vectors vertex = Vectors.GetColumn(d, i); // Берём i-ю вершину (столбец)
                resF[i] = func(model, vertex, @params);
                countFuncEvals++;
            }

            return resF;
        }

        private static void UpdatePoint(Vectors d, Vectors resF, int index, Vectors newPoint, double newValue)
        {
            Vectors.SetColumn(d, newPoint, index);
            resF[index] = newValue;
        }

        private static Vectors CalculateCentroid(Vectors d, int n, int excludeIndex)
        {
            Vectors xc = Vectors.Zeros((1, n));
            for (int j = 0; j <= n; j++)
            {
                if (j != excludeIndex)
                    xc += Vectors.GetColumn(d, j);
            }
            return xc / n;
        }

        private static Vectors ReflectPoint(Vectors xc, Vectors xh, double alpha)
        {
            return (1 + alpha) * xc - alpha * xh;
        }

        private static Vectors ExpandPoint(Vectors xc, Vectors xr, double gamma)
        {
            return (1 - gamma) * xc + gamma * xr;
        }

        private static Vectors ContractPoint(Vectors xc, Vectors xh, double beta)
        {
            return beta * xh + (1 - beta) * xc;
        }

        private static void ShrinkSimplex(
            Vectors d,
            ref IModel model,
            ref Vectors[] @params,
            Vectors resF,
            Vectors xBest,
            LogLikelihoodFunction func,
            ref int countFuncEvals)
        {
            int verticesCount = d.Shape.Item2; // Число вершин (N+1)

            for (int j = 0; j < verticesCount; j++)
            {
                Vectors currentVertex = Vectors.GetColumn(d, j);

                // Пропускаем лучшую вершину
                if (currentVertex.Equals(xBest))
                    continue;

                // Сжатие: x_new = xBest + 0.5*(x_old - xBest)
                Vectors newVertex = xBest + 0.5 * (currentVertex - xBest);

                Vectors.SetColumn(d, newVertex, j); // Обновляем столбец
                resF[j] = func(model, newVertex, @params); // Пересчитываем значение функции
                countFuncEvals++;
            }
        }
    }


    abstract class TestFunction
    {
        public static double Function1(IModel _, Vectors vec, Vectors[] __)
        {
            double tmp = vec[1] - vec[0], tmp2 = 1 - vec[0];
            return 100 * tmp * tmp + tmp2 * tmp2;
        }
        public static double Function2(IModel _, Vectors vec, Vectors[] __)
        {
            double tmp = vec[1] - vec[0] * vec[0], tmp2 = 1 - vec[0];
            return 100 * tmp * tmp + tmp2 * tmp2;
        }
        public static double Function3(IModel _, Vectors vec, Vectors[] __)
        {
            double tmp = (vec[1] - 2) / 3, tmp1 = (vec[1] - 2) / 3, tmp2 = vec[0] - 1, tmp3 = (vec[1] - 1) / 2;
            return -(1 / (1 + tmp * tmp + tmp1 * tmp1) + 3 / (1 + tmp2 * tmp2 + tmp3 * tmp3));
        }
        public static Vectors DFunction1(IModel _, Vectors vec, Vectors[] __)
        {
            return new Vectors([-200 * vec[1] + 202 * vec[0] - 2, 200 * (vec[1] - vec[0])]);
        }
        public static Vectors DFunction2(IModel _, Vectors vec, Vectors[] __)
        {
            var tmp = vec[1] - vec[0] * vec[0];
            return new Vectors([-400 * tmp * vec[0] + 2 * vec[0] - 2, 200 * tmp]);
        }
        public static Vectors DFunction3(IModel _, Vectors vec, Vectors[] __)
        {
            double tmp = (vec[1] - 2) / 3, tmp1 = (vec[1] - 2) / 3, tmp2 = vec[0] - 1, tmp3 = (vec[1] - 1) / 2;
            double denominator1 = 1 + tmp * tmp + tmp1 * tmp1, denominator2 = 1 + tmp2 * tmp2 + tmp3 * tmp3;
            denominator1 *= denominator1; denominator2 *= denominator2;
            return new Vectors([-(2*vec[0] - 4)/(9 * denominator1) - 6*(vec[0] - 1)/denominator2,
                        -(2*vec[1] - 4)/(9 * denominator1) - 3*(vec[1] - 1) / (2*denominator2)]) * -1;
        }
    }
    

}