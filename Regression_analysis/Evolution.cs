
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
        
        public required LogLikelihoodFunction LogLikelihood { get; set; }
        public LogLikelihoodGradient? Gradient { get; set; }
        public required IOprimizator Oprimizator { get; set; }
        public TypeDisribution TypeDisribution { get; set; }
        public bool IsMultiIterationOptimisation { get; set; } = true;
        public int MaxAttempts { get; set; } = 100;
        public Vectors? InitialGuess { get; set; }
        public double Tolerance { get; set; } = 1e-7;
        public int MaxIteration { get; set; } = 1000;


    }

    public class MMKEstimator(MMKConfiguration config) : IParameterEstimator 
    {
        private readonly MMKConfiguration _config = config ?? throw new ArgumentNullException(nameof(config));

        public string Name => "MMK";

        public Vectors EstimateParameters(IModel model, Vectors x, Vectors y, Vectors[] otherParameters) 
        {
            return new Vectors([[]]);   
        }
    }
    


    public static class Test {
        public static void Main(string[] args) {

            DistributionMMKFactory factory = new DistributionMMKFactory();
            factory.Initialize();

            var noramlDist = factory.GetDistribution("Normal");
            /*
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
