
using System.Collections.Concurrent;

using RegressionAnalysisLibrary;

namespace Regression_analysis
{
    public struct ResultEvalueator 
    {
        public Vectors Statistics = new([[]]);
        public required string Distribution;
        public required int CountObservarions;
        public bool IsRound = false;
        public int? RoundDecimals = null;
        public required Vectors ParametersDistribution;

        public ResultEvalueator()
        {
        }

        public override readonly string ToString() => $"SRE {CountObservarions} с ошибкой подчиняющиеся закону: {Distribution}, c параметрами {ParametersDistribution}{(IsRound ? $", округлено до {RoundDecimals} разряда" : "")}";
    }
    
    
    public class RegressionEvaluator
    {
        public static Vectors RoundVector( Vectors vector, int roundDecimals ) 
        {
            for (int i = 0; i < vector.Shape.Item1; i++) {
                for (int j = 0; j < vector.Shape.Item2; j++) {
                    vector[i, j] = double.Round(vector[i, j], roundDecimals);
                }
            }
            return vector;
        }

        public static Vectors GenerateXFromPlan(Vectors planX, Vectors planP, int size, Random? rand) 
        {
            if (!planP.IsVector()) throw new ArgumentException("planP не вектор");
            if (planX.Shape.Item1 != planP.Size) throw new ArgumentException("Количество точек не совпадает с количеством весов");
            planP /= Vectors.Sum(planP);
            int[] counts = new int[planP.Size];
            for (int i = 0; i < counts.Length; i++) {
                counts[i] = (int) Math.Round(planP[i] * size);
            }
            var total = counts.Sum();
            if (total != size) {
                var diff = size - total;
                counts[Vectors.MaxIndex(planP)] += diff;
            }
            Vectors x = Vectors.InitVectors((size, planX.Shape.Item2));
            Vectors row;
            int numrow = 0;
            for (int i = 0; i < counts.Length; i++) {
                row = Vectors.GetRow(planX, i);
                for (int j = 0; j < counts[i]; j++, numrow++) {
                    x.SetRow(row, numrow);
                }
            }
            rand ??= new Random();
            x.ShaffleRows(rand);
            return x;
        }

        public static ResultEvalueator FitParameters(
            IModel model,
            IParameterEstimator evolution,
            int countIteration,
            int countObservations,
            int[] numberParametr,
            IRandomDistribution errorDist,
            Vectors paramsDist,
            IProgress<int> progress,
            CancellationToken token,
            
            Vectors? planX = null,
            Vectors? planP = null,
            Vectors? observations = null,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null,
            bool debug = false,
            bool parallel = false,
            (double start, double end)[]? dimension = null
            )
        {
            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) throw new ArgumentException("Параметры введены не верно.");
            double[][] statistics = new double[numberParametr.Length][];
            for (int i = 0; i < numberParametr.Length; i++)
            {
                if (numberParametr[i] < 0 || numberParametr[i] > model.CountRegressor) throw new ArgumentException("Номер параметра выходит за диапозон доступных");
                statistics[i] = new double[countIteration];
            }
            Vectors x, y, u, matrixX, vectorE, calcTheta;
            var generator = seed is null ? new Random() : new Random((int) seed);
            var localGenerator = new ThreadLocal<Random>(() =>
        seed is null ? new Random() : new Random((int) seed + Thread.CurrentThread.ManagedThreadId));

            int iter = 0;
            object lockObj = new object();

            if (!errorDist.CheckParamsDist(paramsDist)) throw new ArgumentException("Параметры закона распределения введены не верно.");
            if (planX is null)
            {
                if (dimension is null) {
                    var g = new Vectors([-1e+5, 1e+5]);
                    double a = (double) UniformDistribution.Generate(g, generator), b = (double) UniformDistribution.Generate(g, generator);
                    if (a > b) {
                        (a, b) = (b, a);
                    }
                    dimension = [(a, b)];
                }
                x = observations is not null
                    ? observations
                    : LinespaceRandom.Generate((countObservations, model.CountFacts), dimension, generator);
                matrixX = model.CreateMatrixX(x);
                u = (matrixX & model.TrueTheta.T()).T();

                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    Parallel.ForEach(rangePartitioner, new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Environment.ProcessorCount }, range =>
                    {
                        Vectors vectorE, y, calcTheta;
                        int current;
                        for (var i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);

                            for (var j = 0; j < numberParametr.Length; j++)
                                statistics[j][i] = calcTheta[numberParametr[j]];
                            lock (lockObj)
                            {
                                iter++;
                                current = (int) ((double) iter / countIteration * 100);
                            }
                            progress.Report(current);
                        }

                    }
                    );
                }
                else
                {
                    for (int i = 0; i < countIteration; i++)

                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);

#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        for (int j = 0; j < numberParametr.Length; j++)
                            statistics[j][i] = calcTheta[numberParametr[j]];

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
                    }
                }
            }
            else 
            {
                ArgumentNullException.ThrowIfNull(planP);

                if (planX.Shape.Item2 != model.CountFacts) throw new ArgumentException("Число факторов в моделе не совпадает с размерностью точек плана");
                x = GenerateXFromPlan(planX, planP, countObservations, generator);
                matrixX = model.CreateMatrixX(x);
                u = (matrixX & model.TrueTheta.T()).T();

                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    Parallel.ForEach(rangePartitioner, new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Environment.ProcessorCount }, range =>
                    {
                        Vectors vectorE, y, calcTheta;
                        int current;
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                            for (int j = 0; j < numberParametr.Length; j++)
                                statistics[j][i] = calcTheta[numberParametr[j]];
                            lock (lockObj)
                            {
                                iter++;
                                current = (int) ((double) iter / countIteration * 100);
                            }
                            progress.Report(current);
                        }

                    }
                    );
                }
                else
                {
                    for (int i = 0; i < countIteration; i++)
                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        
                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        for (int j = 0; j < numberParametr.Length; j++)
                            statistics[j][i] = calcTheta[numberParametr[j]];

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
                    }
                }
            }
            return new ResultEvalueator() { Statistics = new Vectors(statistics), CountObservarions = countObservations, Distribution = errorDist.Name, ParametersDistribution = paramsDist, IsRound = isRound, RoundDecimals = roundDecimals};

        }

        public static ResultEvalueator Fit(
            IModel model,
            IParameterEstimator evolution,
            int countIteration,
            int countObservations,
            IRandomDistribution errorDist,
            Vectors paramsDist,
            IProgress<int> progress,
            CancellationToken token,
            Vectors? planX = null,
            Vectors? planP = null,
            Vectors? observations = null,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null,
            bool debug = false,
            bool parallel = false,
            (double start, double end)[]? dimension = null
            )
        {
            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) throw new ArgumentException("Параметры введены не верно.");
            double[] statistics = new double[countIteration];
            //Vectors statistics = Vectors.InitVectors((1, countIteration));
            
            
            var localGenerator = new ThreadLocal<Random>(() =>
        seed is null ? new Random() : new Random((int) seed + Thread.CurrentThread.ManagedThreadId));
            Vectors x, y, u, matrixX, vectorE, calcTheta;
            var generator = seed is null ? new Random() : new Random((int) seed);
            
            
            int iter = 0;
            object lockObj = new object();

            if (!errorDist.CheckParamsDist(paramsDist)) throw new ArgumentException("Параметры закона распределения введены не верно.");
            if (planX is null)
            {
                if (dimension is null)
                {
                    var g = new Vectors([-1e+5, 1e+5]);
                    double a = (double) UniformDistribution.Generate(g, generator), b = (double) UniformDistribution.Generate(g, generator);
                    if (a > b)
                    {
                        (a, b) = (b, a);
                    }
                    dimension = [(a, b)];
                }
                x = observations is not null ? observations : Linespace.Generate((countObservations, model.CountFacts), dimension, generator);
                matrixX = model.CreateMatrixX(x);
                u = (matrixX & model.TrueTheta.T()).T();

                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    Parallel.ForEach(rangePartitioner, new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Environment.ProcessorCount }, range =>
                    {
                        Vectors vectorE, y, calcTheta;
                        int current;
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);

                            statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                            lock (lockObj) {
                                iter++;
                                current = (int) ((double) iter / countIteration * 100);
                            }
                            progress.Report(current);
                        }

                    }
                    );
                }
                else
                {
                    for (int i = 0; i < countIteration; i++)

                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);

#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
                    }
                }
            }
            else 
            {
                ArgumentNullException.ThrowIfNull(planP);

                if (planX.Shape.Item2 != model.CountFacts) throw new ArgumentException("Число факторов в моделе не совпадает с размерностью точек плана");
                x = GenerateXFromPlan(planX, planP, countObservations, generator);
                matrixX = model.CreateMatrixX(x);
                u = (matrixX & model.TrueTheta.T()).T();

                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    int current;
                    Parallel.ForEach(rangePartitioner, new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Environment.ProcessorCount }, range =>
                    {
                        Vectors vectorE, y, calcTheta;
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                            statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                            lock (lockObj)
                            {
                                iter++;
                                current = (int) ((double) iter / countIteration * 100);
                            }
                            progress.Report(current);
                        }

                    }
                    );
                }
                else
                {
                    for (int i = 0; i < countIteration; i++)
                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.


                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);

                        progress.Report((int)((double) (i+1) / countIteration * 100));
                    }
                }
            }
            return new ResultEvalueator() { Statistics = new Vectors(statistics), CountObservarions = countObservations, Distribution = errorDist.Name, ParametersDistribution = paramsDist, IsRound = isRound, RoundDecimals = roundDecimals};
        }
        public static Vectors? FitNoParametersDist(
            IModel model,
            IParameterEstimator evolution,
            Func<IModel, Vectors, Vectors[], Vectors> updateParametrs,
            int countIteration,
            int countObservations,
            IRandomDistribution errorDist,
            Vectors paramsDist,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null,
            bool debug = false
            )
        {

            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) return null;
            Vectors statistics = Vectors.InitVectors((1, countIteration));
            var interval = new Vectors([-1e+5, 1e+5]);
            var generator = seed is null ? new Random() : new Random((int) seed);
            (double, double)[] intervalsObservations = new (double, double)[model.CountFacts];
            Vectors x, y, matrixX, vectorE, calcTheta, calcParamDist;
            double stat;
            if (!errorDist.CheckParamsDist(paramsDist)) return null;
            for (int i = 0; i < countIteration; i++)
            {
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
                double a = (double) UniformDistribution.Generate(interval, generator);

                double b = (double) UniformDistribution.Generate(interval, generator);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
                if (a > b)
                    (a, b) = (b, a);
                for (int j = 0; j < model.CountFacts; j++)
                    intervalsObservations[j] = (a, b);
                x = LinespaceRandom.Generate((countObservations, model.CountFacts), intervalsObservations, generator);
                matrixX = model.CreateMatrixX(x);
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                y = (matrixX & model.TrueTheta.T()).T() + vectorE;
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                calcTheta = evolution.EstimateParameters(model, [errorDist.DefaultParametrs, x, y]);
                calcParamDist = updateParametrs(model, calcTheta, [x, y]);
                calcTheta = evolution.EstimateParameters(model, [calcParamDist, x, y]);
                stat = SREСriteria.SRE(model, calcTheta, x, y);
                statistics[i] = stat;
                if (debug)
                {
                    Console.WriteLine($"Итерация: {i}, Статистика: {stat}, Норма: {Vectors.Norm(model.TrueTheta - calcTheta)}");
                }
            }
            return statistics;
        }
    }

    public class EMAlgorithm
    {
        public static (Vectors, Vectors) FitEM(
            IModel model,
            IParameterEstimator evolution,
            Func<IModel, Vectors, Vectors[], Vectors> updateParametrs,
            Vectors initTheta,
            Vectors[] @params,
            double eps = 1e-7,
            int maxIter = 100
            )
        {
            int numIter = 0;
            Vectors x = @params[0], y = @params[1];
            Vectors newTheta = initTheta, oldTheta = double.MaxValue * Vectors.Ones(initTheta.Shape);
            Vectors newParamsDist = updateParametrs(model, initTheta, [x, y]), oldParamsDist = double.MaxValue * Vectors.Ones(newParamsDist.Shape);
            while (Vectors.Norm(newTheta - oldTheta) > eps && numIter++ < maxIter)
            {
                (oldParamsDist, oldTheta) = (newParamsDist, newTheta);
                newTheta = evolution.EstimateParameters(model, [oldParamsDist, x, y]);
                newParamsDist = updateParametrs(model, newTheta, [x, y]);
            }
            Console.WriteLine(Vectors.Norm(newTheta - oldTheta));
            Console.WriteLine(Vectors.Norm(newParamsDist - oldParamsDist));
            Console.WriteLine(numIter);
            return (newTheta, newParamsDist);
        }
    }

    public struct ResultCompareMethods {
        public double[] ParamsMethod1;
        public double[] ParamsMethod2;
        public double[] True_Params;
        public double[][] MatrixX;
        public double[] Y;
    }


    public class ComparisonMethods {
        public static ResultCompareMethods Compare(
            IModel model,
            IParameterEstimator evolution1,
            IParameterEstimator evolution2,
            Vectors[] parameters
            ) 
        {
            var result1 = evolution1.EstimateParameters(model, parameters);
            var result2 = evolution2.EstimateParameters(model, parameters);
            var matrixX = model.CreateMatrixX(parameters[1]);
            return result1 is not null || result2 is not null
                ? new ResultCompareMethods
                {
                    ParamsMethod1 = ((Vectors) result1).ToArrayVector(),
                    ParamsMethod2 = ((Vectors) result2).ToArrayVector(),
                    True_Params = model.TrueTheta.ToArrayVector(),
                    MatrixX = matrixX.ToArray(),
                    Y = parameters[2].ToArrayVector()
                }
                : throw new Exception("Возникли ошибки при оценивании параметров");
        }
    }

}
