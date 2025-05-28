using System.Collections.Concurrent;

namespace RegressionAnalysisLibrary
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

        public static Vectors GenerateXFromPlan(Vectors planX, Vectors planP, int size, Random? rand)
        {
            if (!planP.IsVector()) throw new ArgumentException("planP не вектор");
            if (planX.Shape.Item1 != planP.Size) throw new ArgumentException("Количество точек не совпадает с количеством весов");
            planP /= Vectors.Sum(planP);
            var counts = new int[planP.Size];
            for (var i = 0; i < counts.Length; i++)
                counts[i] = (int) Math.Round(planP[i] * size);
            var total = counts.Sum();
            if (total != size)
            {
                var diff = size - total;
                counts[Vectors.MaxIndex(planP)] += diff;
            }
            var x = Vectors.InitVectors((size, planX.Shape.Item2));
            Vectors row;
            var numrow = 0;
            for (var i = 0; i < counts.Length; i++)
            {
                row = Vectors.GetRow(planX, i);
                for (var j = 0; j < counts[i]; j++, numrow++)
                    x.SetRow(row, numrow);
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
            bool parallel = false,
            (double start, double end)[]? dimension = null
            )
        {
            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) throw new ArgumentException("Параметры введены не верно.");
            var statistics = new double[numberParametr.Length][];
            for (var i = 0; i < numberParametr.Length; i++)
            {
                if (numberParametr[i] < 0 || numberParametr[i] > model.CountRegressor) throw new ArgumentException("Номер параметра выходит за диапозон доступных");
                statistics[i] = new double[countIteration];
            }
            Vectors x, y, u, matrixX, vectorE, calcTheta;
            var generator = seed is null ? new Random() : new Random((int) seed);
            var localGenerator = new ThreadLocal<Random>(() =>
        seed is null ? new Random() : new Random((int) seed + Thread.CurrentThread.ManagedThreadId));

            var iter = 0;
            var lockObj = new object();

            if (!errorDist.CheckParamsDist(paramsDist)) throw new ArgumentException("Параметры закона распределения введены не верно.");
            if (planX is null)
            {
                if (dimension is null)
                {
                    var g = new Vectors([-1e+5, 1e+5]);
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
                    double a = (double) UniformDistribution.Generate(g, generator), b = (double) UniformDistribution.Generate(g, generator);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
                    if (a > b)
                        (a, b) = (b, a);
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
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist, localGenerator.Value);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = Vectors.RoundVector(y, (int) roundDecimals);
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
                    for (var i = 0; i < countIteration; i++)

                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist, generator);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = Vectors.RoundVector(y, (int) roundDecimals);

#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        for (var j = 0; j < numberParametr.Length; j++)
                            statistics[j][i] = calcTheta[numberParametr[j]];

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
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
                        for (var i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist, localGenerator.Value);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = Vectors.RoundVector(y, (int) roundDecimals);
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
                    for (var i = 0; i < countIteration; i++)
                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist, generator);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = Vectors.RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.


                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        for (var j = 0; j < numberParametr.Length; j++)
                            statistics[j][i] = calcTheta[numberParametr[j]];

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
                    }
            }
            return new ResultEvalueator() { Statistics = new Vectors(statistics), CountObservarions = countObservations, Distribution = errorDist.Name, ParametersDistribution = paramsDist, IsRound = isRound, RoundDecimals = roundDecimals };

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
            Criteria typeCriteria = Criteria.SRE,
            Vectors? planX = null,
            Vectors? planP = null,
            Vectors? observations = null,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null,
            bool parallel = false,
            (double start, double end)[]? dimension = null
            )
        {
            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) throw new ArgumentException("Параметры введены не верно.");
            var statistics = new double[countIteration];
            //Vectors statistics = Vectors.InitVectors((1, countIteration));


            var localGenerator = new ThreadLocal<Random>(() =>
        seed is null ? new Random() : new Random((int) seed + Thread.CurrentThread.ManagedThreadId));
            Vectors x, y, u, matrixX, vectorE, calcTheta;
            var generator = seed is null ? new Random() : new Random((int) seed);


            var iter = 0;
            var lockObj = new object();

            if (!errorDist.CheckParamsDist(paramsDist)) throw new ArgumentException("Параметры закона распределения введены не верно.");
            if (planX is null)
            {
                if (dimension is null)
                {
                    var g = new Vectors([-1e+5, 1e+5]);
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
                    double a = (double) UniformDistribution.Generate(g, generator), b = (double) UniformDistribution.Generate(g, generator);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
                    if (a > b)
                        (a, b) = (b, a);
                    dimension = [(a, b)];
                }
                x = observations is not null ? observations : LinespaceRandom.Generate((countObservations, model.CountFacts), dimension, generator);
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
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist, localGenerator.Value);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = Vectors.RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);

                            switch (typeCriteria)
                            {
                                case Criteria.SRE:
                                    statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                                    break;
                                case Criteria.LR:
                                    statistics[i] = evolution is MMPEstimator mmp
                                        ? LRCriteria.CalculateLR(mmp.Config.Functions.LogLikelihood, model, calcTheta, [paramsDist, x, y])
                                        : throw new ArgumentException("Данный критерий применим только для ММП-оценок");
                                    break;
                                default:
                                    throw new ArgumentException("Введён неизвестный критерий");
                            }

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
                    for (var i = 0; i < countIteration; i++)

                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist, generator);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = Vectors.RoundVector(y, (int) roundDecimals);

#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        switch (typeCriteria)
                        {
                            case Criteria.SRE:
                                statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                                break;
                            case Criteria.LR:
                                statistics[i] = evolution is MMPEstimator mmp
                                    ? LRCriteria.CalculateLR(mmp.Config.Functions.LogLikelihood, model, calcTheta, [paramsDist, x, y])
                                    : throw new ArgumentException("Данный критерий применим только для ММП-оценок");
                                break;
                            default:
                                throw new ArgumentException("Введён неизвестный критерий");
                        }

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
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
                        for (var i = range.Item1; i < range.Item2; i++)
                        {
                            token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist, localGenerator.Value);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = Vectors.RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                            switch (typeCriteria)
                            {
                                case Criteria.SRE:
                                    statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                                    break;
                                case Criteria.LR:
                                    statistics[i] = evolution is MMPEstimator mmp
                                        ? LRCriteria.CalculateLR(mmp.Config.Functions.LogLikelihood, model, calcTheta, [paramsDist, x, y])
                                        : throw new ArgumentException("Данный критерий применим только для ММП-оценок");
                                    break;
                                default:
                                    throw new ArgumentException("Введён неизвестный критерий");
                            }
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
                    for (var i = 0; i < countIteration; i++)
                    {
                        token.ThrowIfCancellationRequested();
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist, generator);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = Vectors.RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.


                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        switch (typeCriteria)
                        {
                            case Criteria.SRE:
                                statistics[i] = SREСriteria.SRE(model, calcTheta, x, y);
                                break;
                            case Criteria.LR:
                                statistics[i] = evolution is MMPEstimator mmp
                                    ? LRCriteria.CalculateLR(mmp.Config.Functions.LogLikelihood, model, calcTheta, [paramsDist, x, y])
                                    : throw new ArgumentException("Данный критерий применим только для ММП-оценок");
                                break;
                            default:
                                throw new ArgumentException("Введён неизвестный критерий");
                        }

                        progress.Report((int) ((double) (i + 1) / countIteration * 100));
                    }
            }
            return new ResultEvalueator() { Statistics = new Vectors(statistics), CountObservarions = countObservations, Distribution = errorDist.Name, ParametersDistribution = paramsDist, IsRound = isRound, RoundDecimals = roundDecimals };
        }
    }
}
