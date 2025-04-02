
namespace Regression_analysis
{
    public interface IParameterEstimator 
    { 
        string Name { get; }
        public abstract Vectors EstimateParameters(IModel model, Vectors x, Vectors y, Vectors[] otherParameters);
    }

    public class MNKEstimator : IParameterEstimator {
        public string Name => "MNK";

        public Vectors EstimateParameters(IModel model, Vectors x, Vectors y, Vectors[] otherParameters) 
        {
            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Вектор 'x' не соответствует количеству факторов в моделе 'model'");

            if (!y.IsVector() || y.Size != x.Shape.Item1)
                throw new ArgumentException("Вектор 'y' имеет не правильный вид.");

            var matrixX = model.CreateMatrixX(x);

            Vectors estimatedTheta;
            try
            {
                var tMatrixX = matrixX.T();
                estimatedTheta = (Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & y.T();
            }
            catch
            {
                throw new Exception("Не правильные данные");
            }
            return estimatedTheta;
        }
    }

    public class MMKConfiguration 
    {
        public delegate double LogLikelihoodFunction(IModel model, Vectors @params, Vectors[] parametrs);

        public delegate Vectors LogLikelihoodGradient(IModel model, Vectors @params, Vectors[] parametrs);

        public required LogLikelihoodFunction LogLikelihood { get; set; }
        public required LogLikelihoodGradient Gradient { get; set; }
        public Vectors? InitialGuess { get; set; }
        public double Tolerance { get; set; } = 1e-7;
        public int MatIteration { get; set; } = 1000;

    }

    public class MMKEstimator(MMKConfiguration config) : IParameterEstimator 
    {
        private readonly MMKConfiguration _config = config ?? throw new ArgumentNullException(nameof(config));

        public string Name => "MMK";

        public Vectors EstimateParameters(IModel model, Vectors x, Vectors y, Vectors[] otherParameters) 
        { 
            
        
        }
    }

    public class MNK
    {
        public static Vectors EstimateParametrs(IModel model, Vectors x, Vectors y, Vectors? matrixX = null) {

            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(x);
            ArgumentNullException.ThrowIfNull(y);

            matrixX ??= model.CreateMatrixX(x);

            Vectors estimatedTheta;
            try
            {
                var tMatrixX = matrixX.T();
                estimatedTheta = (Vectors.Inv(tMatrixX & matrixX) & tMatrixX) & y.T();
            }
            catch {
                throw new Exception("Error. Incorrect Data.");
            }
            return estimatedTheta;
        }
    }

    public class MMK
    {

        public static Vectors EstimateParametrsNormalDist(IModel model, Vectors x, Vectors y, Vectors? matrixX = null) {
            return MNK.EstimateParametrs(model, x, y, matrixX);
        }

        private static Vectors CheckArguments(Vectors parametr) {
            if (parametr.IsVector()) {
                if (parametr.Shape.Item2 < parametr.Shape.Item1)
                    parametr = parametr.T();
                if (parametr.Shape.Item1 != 1)
                    throw new ArgumentException("params имеет не тот размер");
            }
            else {
                throw new ArgumentException("params не является вектором");
            }
            return parametr;
        }

        public static double LogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs) {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            @paramsDist = CheckArguments(@paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (@paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {@paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0] - @paramsDist[0];
                residuals -= Math.Log(1 + Math.Pow(funcVector / paramsDist[1], 2));
            }
            return residuals;
        }

        public static Vectors DfLogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            var residuals = Vectors.Zeros((1, model.CountRegressor));

            double funcVector;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = y[i] - (vector & @params)[0] - paramsDist[0];
                residuals += (funcVector / (paramsDist[1] * paramsDist[1] + funcVector * funcVector)) * vector;
            }
            return 2 * residuals;
        }

        public static Vectors GessianLogLikelihoodChoshiDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            var residuals = Vectors.Zeros((model.CountRegressor, model.CountRegressor));

            double funcVector;
            Vectors vector;
            for (var i = 0; i < y.Size; i++)
            {
                vector = model.VectorFunc(Vectors.GetRow(x, i));
                funcVector = Math.Pow(y[i] - (vector & @params)[0] - paramsDist[0], 2);
                var value = paramsDist[1] * paramsDist[1];
                residuals += ((funcVector - value) / (value + funcVector)) * (vector.T() & vector);
            }
            return 2 * residuals;
        }

        public static double LogLikelihoodExponentialDist(IModel model, Vectors @params, params Vectors[] parametrs) {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            double funcVector;
            double penaity = 0.0;
            for (var i = 0; i < y.Size; i++) {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0];
                residuals -= funcVector - paramsDist[0];
                penaity += Math.Pow(double.Abs(funcVector) - funcVector, 2) / 2;
            }
            return (1 / @paramsDist[1]) * residuals - penaity;
        }

        public static Vectors DfLogLikelihoodExponentialDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");

            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            paramsDist = CheckArguments(paramsDist);
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals += funcVector;
            }
            return (1 / @paramsDist[1]) * residuals;
        }
        public static double LogLikelihoodLaplaceDist(IModel model, Vectors @params, params Vectors[] parametrs) {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            double residuals = 0.0;

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++) {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= Math.Abs(y[i] - (funcVector & @params)[0] - paramsDist[0]);
            }

            return paramsDist[1] * residuals;
        }

        public static Vectors DfLogLikelihoodLaplaceDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 2)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 2");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                residuals -= Math.Sign(y[i] - (funcVector & @params)[0] - paramsDist[0]) * funcVector;
            }

            return paramsDist[1] * residuals;
        }

        public static double LogLikelihoodGammaDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 3)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 3");

            double residuals = 0.0;

            double funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0] - paramsDist[0];
                residuals += (@paramsDist[2] - 1) * Math.Log(funcVector) - funcVector / paramsDist[1];
            }

            return residuals;
        }

        public static Vectors DfLogLikelihoodGammaDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 3)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 3.");
            var @paramsDist = parametrs[0];
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            paramsDist = CheckArguments(paramsDist);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();
            if (paramsDist.Size != 3)
                throw new ArgumentException($"Количетво элементов не соответствует количеству параметров в распределении. Параметров задано {paramsDist.Size}, а нужно 3");

            Vectors residuals = Vectors.Zeros((1, model.CountRegressor));

            Vectors funcVector;
            for (var i = 0; i < y.Size; i++)
            {
                funcVector = model.VectorFunc(Vectors.GetRow(x, i));
                var e = y[i] - (funcVector & @params)[0] - paramsDist[0];
                residuals += (1 / paramsDist[1] - (paramsDist[2] - 1) / e) * funcVector;
            }

            return residuals;
        }
        public static double LogLikelihoodUniformDist(IModel model, Vectors @params, params Vectors[] parametrs)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (parametrs.Length < 2)
                throw new ArgumentException($"Полученно параметров {parametrs.Length}, а ожидалось 2.");
            var x = parametrs[1];
            var y = parametrs[2];

            if (x.Shape.Item2 != model.CountFacts)
                throw new ArgumentException("Размерность строк 'x' не соответсвует количеству факторов модели");
            y = CheckArguments(y);
            @params = CheckArguments(@params);
            if (@params.Size != model.CountRegressor)
                throw new ArgumentException("Количество параметров не соответствует с количеством регрессор в модели");
            @params = @params.T();

            double maxEps = double.MinValue;
            double minEps = double.MaxValue;

            double funcVector;
            var n = y.Size;
            for (var i = 0; i < n; i++)
            {
                funcVector = y[i] - (model.VectorFunc(Vectors.GetRow(x, i)) & @params)[0];
                if (funcVector > maxEps)
                    maxEps = funcVector;
                if (funcVector < minEps)
                    minEps = funcVector;
            }
            return minEps - maxEps;
        }
    }


    public static class Test {
        public static void Main(string[] args) {

            
            
            Vectors TrueTheta = new Vectors([5, 2, 7, 5]);
            LiniarModel model1 = new LiniarModel(3, [], TrueTheta, true);
            Vectors X_test = LinespaceRandom.Generate((250, 3), [(-1, 1), (-1, 1), (-1, 1)], new Random());
            Vectors MatrixX = model1.CreateMatrixX(X_test);
            var VectorE = NormalDistribution.Generate((1, 250), new Vectors([0, 10]), new Random());
            var Y = (MatrixX & model1.TrueTheta.T()).T() + VectorE;
            LiniarModel model2 = new LiniarModel(3, [(0, 1), (1, 2), (0, 2)], new Vectors([1,1,1,1,1,1]), false);
            var CalcTheta = MNK.EstimateParametrs(model2, X_test, Y);
            var CalcTheta2 = MNK.EstimateParametrs(model1, X_test, Y);
            Console.WriteLine(CalcTheta);
            Console.WriteLine(CalcTheta2);
            
            Console.WriteLine(SREСriteria.SRE(model2, CalcTheta, X_test, Y));
            Console.WriteLine(SREСriteria.SRE(model1, CalcTheta2, X_test, Y));
            /*
            var VectorE = ExponentialDistribution.Generate((1, 1000), new Vectors([20, 2]), new Random());
            #pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
            var Y = (MatrixX & model1.TrueTheta.T()).T() + VectorE;
            #pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.


            var OldCalcTheta = Vectors.Zeros((1, model1.CountRegressor));
            var CalcTheta = Vectors.Ones((1, model1.CountRegressor));
            double Loc = Vectors.Min(Y);
            double oldScale = 0;
            double Scale = 1;
            double eps = 1e-7;
            Vectors E = new Vectors([[]]);
            OptResExtended Result;

            while (Vectors.Norm(CalcTheta - OldCalcTheta) > eps && double.Abs(Scale - oldScale) > eps) {

                // Получаем приближённое значение Theta для текущих loc и scale
                Result = NelderMeadOptimizer.OptimisateRandomInit(
                    (IModel model, Vectors vector, Vectors[] Params) => -MMK.LogLikelihoodExponentialDist(model, vector, Params),
                    model1,
                    [new Vectors([Loc, Scale]), X_test, Y],
                    (1, model1.CountRegressor),
                    1e-12
                    //MinValue: 0
                );
                (OldCalcTheta, CalcTheta) = (CalcTheta, Result.MinPoint);
                
                // Получаем приближённое значение смещения
                
                // Обновляем значение маштаба
                E = Y - (MatrixX & CalcTheta.T()).T();
                Loc = Vectors.Min(E);
                //Loc = Vectors.Min(E);
                (oldScale, Scale) = (Scale, Vectors.Sum(E)/E.Size);
            }
            Console.WriteLine(Vectors.Min(Y));
            Y -= Loc;
            Result = NelderMeadOptimizer.OptimisateRandomInit(
                    (IModel model, Vectors vector, Vectors[] Params) => -MMK.LogLikelihoodExponentialDist(model, vector, Params),
                    model1,
                    [new Vectors([0, Scale]), X_test, Y],
                    (1, model1.CountRegressor),
                    1e-12
                //MinValue: 0
            );
            CalcTheta = Result.MinPoint;

            

            Console.WriteLine($"Полученные значения Theta: {CalcTheta} - norm({Vectors.Norm(CalcTheta - TrueTheta)})");
            Console.WriteLine($"Полученные значения сдвига: {Loc} - norm({double.Abs(Loc)})");
            Console.WriteLine($"Полученные значения маштаба: {Scale} - norm({double.Abs(Scale - 14)})");
            
            int seed = 21;
            double avg1 = 0.0;
            double avg2 = 0.0;
            long ts = 0;
            
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            for (int i = 0; i < 2; i++) {
                VectorE = dist.Exponential(loc: 0, scale: 45.2, Shape: (1, 1000));
                Y = (MatrixX & model1.TrueTheta.T()).T() + VectorE;
                ts = stopwatch.ElapsedMilliseconds;
                var Result = NelderMeadOptimizer.OptimisateRandomInit(
                    (IModel model, Vectors vector, Vectors[] Params) => -MMK.LogLikelihoodExponentialDist(model, vector, Params),
                    model1,
                    [new Vectors([0, 45.2]), X_test, Y],
                    (1, model1.CountRegressor),
                    1e-7,
                    MinValue: 0,
                    seed: seed
                );
                var temp = stopwatch.ElapsedMilliseconds - ts;
                ts = stopwatch.ElapsedMilliseconds;
                var Result2 = NelderMeadOptimizer.OptimisateRandomInit(
                    (IModel model, Vectors vector, Vectors[] Params) => -MMK.LogLikelihoodExponentialDist(model, vector, Params),
                    model1,
                    [new Vectors([0, 45.2]), X_test, Y],
                    (1, model1.CountRegressor),
                    1e-16,
                    MinValue: 0,
                    seed: seed
                );
                Console.WriteLine($"{i} - {temp} - {stopwatch.ElapsedMilliseconds - ts} - {Vectors.Norm(Result.MinPoint - TrueTheta)} - {Vectors.Norm(Result2.MinPoint - TrueTheta)}");
                avg1 += Vectors.Norm(Result.MinPoint - TrueTheta);
                avg2 += Vectors.Norm(Result2.MinPoint - TrueTheta);
            }
            stopwatch.Stop();
            Console.WriteLine($"Всего потраченно времени: {stopwatch.ElapsedMilliseconds}");
            avg2 /= 100;
            avg1 /= 100;
            Console.WriteLine(avg1);
            Console.WriteLine(avg2);

            var stream = new StringBuilder();
            stream.AppendLine("Экспоненциальное распределение (0, 45.2)");
            stream.AppendLine($"0 {X_test.Shape.Item1}");
            for (int i = 0; i < X_test.Shape.Item1; i++) {
                stream.AppendLine(FormattableString.Invariant($"{Y[i]}"));
            }
            File.WriteAllText("out.dat", stream.ToString());
            */
        }
    }

}
