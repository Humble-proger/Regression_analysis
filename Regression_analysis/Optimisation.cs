
using Regression_analysis;

namespace RegressionAnalysisLibrary
{

    public interface IOprimizator
    {
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
        public int NumIteration;
        public double Tol;
        public int CountCalcFunc;
        public int CountCalcGradient;
        public int CountCalcHessian;
        public bool Convergence;
        public double? Norm;
        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\nКоличество подсчёта градиента: {CountCalcGradient};\nКоличество подсчёта гессиана: {CountCalcHessian};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {Tol}";
        }
    }
    public struct OptResExtended
    {
        public Vectors MinPoint;
        public int NumIteration;
        public double Tol;
        public int CountCalcFunc;
        public int CountCalcGradient;
        public int CountCalcHessian;
        public bool Convergence;
        public double? Norm;
        public Vectors InitParametrs;
        public int NumberRebounds;

        public override readonly string ToString()
        {
            return $"Результат оптимизации:\nНайденный минимум: {MinPoint};\nНачальные значения: {InitParametrs};\nКоличество подбора начальных значений: {NumberRebounds};\nКоличество итераций: {NumIteration};\nКоличество подсчёта функции: {CountCalcFunc};\nКоличество подсчёта градиента: {CountCalcGradient};\nКоличество подсчёта гессиана: {CountCalcHessian};\n{(Norm != null ? $"Полученная норма: {Norm};\n" : "")}Сошлось: {(Convergence ? "Да" : "Нет")};\nТочность: {Tol}";
        }
    }

    public static class QuadraticInterpolation
    {
        //private static double Min(params double[] numbers) => numbers.Min();
        public static OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient grad,
            IModel model,
            Vectors[] @params,
            Vectors xk,
            Vectors pk,
            double initialAlpha,
            double eps = 1e-4,
            int maxIter = 20
            )
        {
            var result = new OptRes()
            {
                CountCalcFunc = 2,
                CountCalcGradient = 1,
                NumIteration = 0,
                Tol = eps,
                Convergence = false
            };
            // Начальные точки для интерполяции
            double alpha0 = 0;

            Vectors Gradient(Vectors x) => grad(model, x, @params);
            double Function(Vectors x) => func(model, x, @params);

            var f0 = Function(xk);
            var g0 = Gradient(xk).Dot(pk.T())[0];

            if (g0 >= 0)
                throw new ArgumentException("Direction is not a descent direction");

            var alpha1 = initialAlpha;
            var f1 = Function(xk + pk * alpha1);

            var iter = 0;
            while (iter < maxIter)
            {
                // Квадратичная интерполяция
                var alpha = QuadraticInterpolationStep(alpha0, f0, g0, alpha1, f1);

                // Ограничение максимального шага
                alpha = Math.Min(alpha, 10 * alpha1);

                // Проверка условия Армихо
                var fAlpha = Function(xk + pk * alpha);
                result.CountCalcFunc++;
                if (fAlpha <= f0 + eps * alpha * g0)
                {
                    result.Convergence = true;
                    result.MinPoint = new Vectors([alpha]);
                    return result;
                }

                // Подготовка к следующей итерации
                if (fAlpha < f1 || alpha1 == initialAlpha)
                {
                    alpha0 = alpha1;
                    f0 = f1;
                    alpha1 = alpha;
                    f1 = fAlpha;
                }
                else
                {
                    alpha1 = alpha;
                    f1 = fAlpha;
                }

                iter++;
            }

            result.MinPoint = new Vectors([alpha1]);
            return result;
        }

        private static double QuadraticInterpolationStep(double alpha0, double f0, double g0,
                                               double alpha1, double f1)
        {
            // Квадратичная интерполяция между alpha0 и alpha1
            var denominator = 2 * (f1 - f0 - g0 * (alpha1 - alpha0));
            if (denominator <= 0)
                return (alpha0 + alpha1) / 2; // Если интерполяция неудачна, берем середину

            return alpha0 - g0 * Math.Pow(alpha1 - alpha0, 2) / denominator;
        }
    }

    public static class GoldenSection
    {
        private const double Phi = 1.618033988749895; // Золотое сечение (1 + sqrt(5))/2

        public static OptRes Optimize(
            LogLikelihoodFunction func,
            IModel model,
            Vectors[] @params,
            Vectors xk,
            Vectors pk,
            double eps = 1e-7,
            int maxIter = 100,
            double minalpha = 0,
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
                result.CountCalcFunc++;
                return result;
            }

            var a = minalpha;
            var b = maxalpha;

            var x1 = b - (b - a) / Phi;
            var x2 = a + (b - a) / Phi;

            var f1 = Func(x1); result.CountCalcFunc++;
            var f2 = Func(x2); result.CountCalcFunc++;

            for (; result.NumIteration < maxIter; result.NumIteration++)
            {
                if (Math.Abs(b - a) < eps)
                {
                    result.MinPoint = new Vectors([(a + b) / 2]);
                    result.CountCalcFunc++;
                    result.Convergence = true;
                    return result;
                }

                if (f1 < f2)
                {
                    b = x2;
                    x2 = x1;
                    f2 = f1;
                    x1 = b - (b - a) / Phi;
                    f1 = Func(x1); result.CountCalcFunc++;
                }
                else
                {
                    a = x1;
                    x1 = x2;
                    f1 = f2;
                    x2 = a + (b - a) / Phi;
                    f2 = Func(x2); result.CountCalcFunc++;
                }
            }

            result.MinPoint = new Vectors([(a + b) / 2]);
            result.CountCalcFunc++;
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

            Vectors Gradient(Vectors x) => dffunc(model, x, @params);
            var n = x0.Size;
            var xk = x0.Clone();
            var gfk = Gradient(xk);
            var pk = -gfk;

            var result = new OptRes()
            {
                CountCalcFunc = 0,
                CountCalcGradient = 1,
                Norm = Vectors.Norm(gfk),
                NumIteration = 0
            };
            var stepsSinceReset = 0;

            while (result.Norm > eps && result.NumIteration < maxIter)
            {
                //OptRes linResult = QuadraticInterpolation.Optimisate(func, dffunc, model, @params, xk, pk, 1.0);

                //result.CountCalcGradient += linResult.CountCalcGradient;
                //result.CountCalcFunc += linResult.CountCalcFunc;
                var alpha = LineSearch(func, dffunc, model, @params, xk, pk);

                var xkp1 = xk + alpha * pk;
                result.Norm = Vectors.Norm(xkp1 - xk);
                var gfkp1 = Gradient(xkp1);

                stepsSinceReset++;
                if (stepsSinceReset < 2 * n)
                {
                    var w = ScalarMult(gfkp1, gfkp1) / double.Max(ScalarMult(gfk, gfk), double.Epsilon);
                    pk = -gfkp1 + w * pk;
                }
                else
                {
                    pk = -gfkp1;
                    stepsSinceReset = 0;
                }
                (xk, gfk) = (xkp1, gfkp1);
                result.NumIteration++;
            }

            result.Convergence = result.Norm < eps;
            result.MinPoint = xk;
            return result;
        }


        private double LineSearch(
            LogLikelihoodFunction f,
            LogLikelihoodGradient grad,
            IModel model,
            Vectors[] parameters,
            Vectors x,
            Vectors d,
            int maxIter = 20,
            double tol = 1e-4
            )
        {
            var alpha = 1.0;
            var c = 0.1; // Параметр для условия Армихо
            var rho = 0.5; // Коэффициент уменьшения шага

            Vectors Gradient(Vectors x) => grad(model, x, parameters);
            double Function(Vectors x) => f(model, x, parameters);


            var f0 = Function(x);
            var g0 = Gradient(x);
            var slope0 = g0.Dot(d.T())[0];

            for (var i = 0; i < maxIter; i++)
            {
                var xNew = x + d * alpha;
                var fAlpha = Function(xNew);

                // Квадратичная интерполяция
                if (i > 0 && fAlpha > f0 + c * alpha * slope0)
                {
                    var alphaQuad = -slope0 * alpha * alpha / (2 * (fAlpha - f0 - slope0 * alpha));
                    alpha = Math.Max(alphaQuad, alpha * rho);
                }

                if (fAlpha <= f0 + c * alpha * slope0)
                    return alpha;

                alpha *= rho;
            }

            return alpha;
        }

        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
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
                result.InitParametrs = rand.Generate((1, model.CountRegressor), tmp);
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
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
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
            }
            return result;
        }
    }

    public class DFPOptimizator : IOprimizator
    {
        public string Name => "DFP";

        public bool IsUseGradient => true;

        public OptRes Optimisate(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? grad,
            IModel model,
            Vectors initialGuess,
            Vectors[] parameters,
            double tol = 1e-7,
            int maxIter = 1000
            )
        {
            if (!initialGuess.IsVector()) throw new Exception("InitParam должен быть вектором.");
            if (grad is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");

            var n = initialGuess.Size;
            var x = initialGuess.Clone();

            Vectors Gradient(Vectors x) => grad(model, x, parameters);

            var g = Gradient(x);
            var H = Vectors.Eig((n, n));

            var result = new OptRes()
            {
                CountCalcFunc = 0,
                NumIteration = 0,
                Tol = tol,
                Norm = Vectors.Norm(g),
                MinPoint = initialGuess,
            };

            while (result.NumIteration < maxIter && result.Norm > tol)
            {
                var d = -(H & g.T()).T(); // Направление спуска

                // Линейный поиск с квадратичной интерполяцией
                var linSearch = QuadraticInterpolation.Optimisate(func, grad, model, parameters, x, d, 1.0);

                result.CountCalcFunc += linSearch.CountCalcFunc;
                result.CountCalcGradient += linSearch.CountCalcGradient;
                var alpha = linSearch.MinPoint[0];

                var xNew = Vectors.Add(x, d * alpha);
                var gNew = Gradient(xNew);
                result.CountCalcGradient++;

                var deltaX = xNew - x;
                var deltaG = gNew - g;

                result.Norm = Vectors.Norm(deltaX);

                // Обновление приближения Гессиана по формуле DFP
                H = UpdateHessianApproximationDFP(H, deltaX, deltaG);

                x = xNew;
                g = gNew;

                result.NumIteration++;
            }

            result.Convergence = result.Norm < tol;
            result.MinPoint = x;
            return result;
        }

        private static Vectors UpdateHessianApproximationDFP(Vectors H, Vectors deltaX, Vectors deltaG)
        {
            var n = deltaX.Size;
            var HNew = H.Clone();

            var HdeltaG = (H & deltaG.T()).T();
            var deltaGtHdeltaG = deltaG.Dot(HdeltaG.T())[0];
            var deltaXtDeltaG = deltaX.Dot(deltaG.T())[0];

            // Проверка условия кривизны
            if (deltaXtDeltaG <= 0)
                return H; // Не обновляем H если условие кривизны не выполняется

            // Первое слагаемое DFP
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                    HNew[i, j] += deltaX[i] * deltaX[j] / deltaXtDeltaG;

            // Второе слагаемое DFP
            for (var i = 0; i < n; i++)
                for (var j = 0; j < n; j++)
                    HNew[i, j] -= HdeltaG[i] * HdeltaG[j] / deltaGtHdeltaG;

            return HNew;
        }

        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            )
        {
            if (dffunc is null) throw new ArgumentException($"В {Name} для оптимизации используется градиент для оптимизации, но dffunc является null");
            var rand = seed == null ? new() : (UniformDistribution) new(seed);
            OptRes resMethod;
            var tmp = new Vectors([minValue, maxValue]);
            OptResExtended result = new()
            {
                Tol = eps,
                InitParametrs = x0 ?? Vectors.Zeros((1, model.CountRegressor)),
            };
            resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
            result.Convergence = resMethod.Convergence;
            result.NumIteration = resMethod.NumIteration;
            result.NumberRebounds += 1;
            result.Norm = resMethod.Norm;
            result.MinPoint = resMethod.MinPoint;
            result.CountCalcFunc = resMethod.CountCalcFunc;
            for (; result.NumberRebounds < maxIter && !result.Convergence; result.NumberRebounds++)
            {
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                result.InitParametrs = rand.Generate((1, model.CountRegressor), tmp);
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
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
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
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

            var n = x0.Shape.Item2;
            if (n < 2)
                throw new ArgumentException("Dimension must be at least 2");

            // Инициализация симплекса (каждый СТОЛБЕЦ D — вершина размерности N)
            var d = InitializeSimplex(x0, n, T);
            var resF = EvaluateSimplex(func, ref model, ref @params, d, ref result.CountCalcFunc);

            while (result.NumIteration < maxIter)
            {
                var indexMin = Vectors.MinIndex(resF);
                var indexMax = Vectors.MaxIndex(resF);
                var indexSecondMax = Vectors.SecondMaxIndex(resF, indexMax);

                // Критерий останова
                if ((result.Norm = Vectors.NormDifference(resF, resF[indexMin])) < eps)
                    break;

                // Центр тяжести (исключая худшую вершину)
                var xc = CalculateCentroid(d, n, indexMax);

                // Отражение
                var xr = ReflectPoint(xc, Vectors.GetColumn(d, indexMax), Alpha);
                var fr = func(model, xr, @params);
                result.CountCalcFunc++;

                if (fr < resF[indexMin])
                {
                    // Растяжение
                    var xe = ExpandPoint(xc, xr, Gamma);
                    var fe = func(model, xe, @params);
                    result.CountCalcFunc++;
                    UpdatePoint(d, resF, indexMax, fe < fr ? xe : xr, Math.Min(fe, fr));
                }
                else if (fr < resF[indexSecondMax])
                    UpdatePoint(d, resF, indexMax, xr, fr);
                else
                {
                    // Сжатие
                    var xs = ContractPoint(xc, Vectors.GetColumn(d, indexMax), Beta);
                    var fs = func(model, xs, @params);
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
            return result;
        }

        public OptResExtended OptimisateRandomInit(
            LogLikelihoodFunction func,
            LogLikelihoodGradient? dffunc,
            IModel model,
            Vectors[] @params,
            double eps,
            double minValue = -10,
            double maxValue = 10,
            int maxIter = 100,
            int? seed = null,
            Vectors? x0 = null
            )
        {
            var rand = seed == null ? new() : (UniformDistribution) new(seed);
            OptRes resMethod;
            var tmp = new Vectors([minValue, maxValue]);
            OptResExtended result = new()
            {
                Tol = eps,
                InitParametrs = x0 ?? Vectors.Zeros((1, model.CountRegressor)),
            };
            resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
            result.Convergence = resMethod.Convergence;
            result.NumIteration = resMethod.NumIteration;
            result.NumberRebounds += 1;
            result.Norm = resMethod.Norm;
            result.MinPoint = resMethod.MinPoint;
            result.CountCalcFunc = resMethod.CountCalcFunc;
            for (; result.NumberRebounds < maxIter && !result.Convergence; result.NumberRebounds++)
            {
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                result.InitParametrs = rand.Generate((1, model.CountRegressor), tmp);
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                resMethod = Optimisate(func, dffunc, model, result.InitParametrs, @params, eps);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                if (resMethod.Convergence)
                {
                    result.Convergence = true;
                    result.MinPoint = resMethod.MinPoint;
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
                        result.NumIteration = resMethod.NumIteration;
                    }
                    result.CountCalcFunc += resMethod.CountCalcFunc;
                }
            }
            return result;
        }

        private static Vectors InitializeSimplex(Vectors x0, int n, double t)
        {
            var d = Vectors.InitVectors((n, n + 1));
            var d1 = t * (Math.Sqrt(n + 1) + n - 1) / (n * Math.Sqrt(2));
            var d2 = t * (Math.Sqrt(n + 1) - 1) / (n * Math.Sqrt(2));

            // Первая вершина — x0
            for (var i = 0; i < n; i++)
                d[i, 0] = x0[i];

            // Остальные вершины
            for (var i = 0; i < n; i++)
                for (var j = 1; j <= n; j++)
                    d[i, j] = i == j - 1 ? x0[i] + d1 : x0[i] + d2;
            return d;
        }

        private static Vectors EvaluateSimplex(LogLikelihoodFunction func, ref IModel model, ref Vectors[] @params, Vectors d, ref int countFuncEvals)
        {
            var verticesCount = d.Shape.Item2; // Число столбцов (N+1 вершин)

            var resF = Vectors.Zeros((1, verticesCount)); // Вектор значений функции

            for (var i = 0; i < verticesCount; i++)
            {
                var vertex = Vectors.GetColumn(d, i); // Берём i-ю вершину (столбец)
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
            var xc = Vectors.Zeros((1, n));
            for (var j = 0; j <= n; j++)
                if (j != excludeIndex)
                    xc += Vectors.GetColumn(d, j);
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
            var verticesCount = d.Shape.Item2; // Число вершин (N+1)

            for (var j = 0; j < verticesCount; j++)
            {
                var currentVertex = Vectors.GetColumn(d, j);

                // Пропускаем лучшую вершину
                if (currentVertex.Equals(xBest))
                    continue;

                // Сжатие: x_new = xBest + 0.5*(x_old - xBest)
                var newVertex = xBest + 0.5 * (currentVertex - xBest);

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