﻿using System;
using System.Collections.Concurrent;

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
        
        
        public static ResultEvalueator Fit(
            IModel model,
            IParameterEstimator evolution,
            int countIteration,
            int countObservations,
            IRandomDistribution errorDist,
            Vectors paramsDist,
            Vectors? planX = null,
            Vectors? planP = null,
            bool isRound = false,
            int? roundDecimals = null,
            int? seed = null,
            bool debug = false,
            bool parallel = false
            )
        {

            var progress = new ProgressBarConsole() { CountIteration = countIteration };
            if (countIteration <= 0 || countObservations <= 0 || paramsDist.Size != errorDist.CountParametrsDistribution) throw new ArgumentException("Параметры введены не верно.");
            double[] statistics = new double[countIteration];
            //Vectors statistics = Vectors.InitVectors((1, countIteration));
            
            
            var localGenerator = new ThreadLocal<Random>(() =>
        seed is null ? new Random() : new Random((int) seed + Thread.CurrentThread.ManagedThreadId));
            Vectors x, y, matrixX, vectorE, calcTheta;
            double stat, avg_norm = 0.0, avg_stat = 0.0;
            var generator = seed is null ? new Random() : new Random((int) seed);
            
            
            int iter = 0;
            object lockObj = new object();

            if (!errorDist.CheckParamsDist(paramsDist)) throw new ArgumentException("Параметры закона распределения введены не верно.");
            if (planX is null)
            {
                var interval = new Vectors([-1e+5, 1e+5]);
                double[][] ab = new double[2][];
                if (parallel)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        ab[i] = new double[countIteration];
                        Parallel.For(0, countIteration, j => {
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            ab[i][j] = (double) UniformDistribution.Generate(interval, localGenerator.Value);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
                        });
                    }
                }
                else
                {
                    for (int i = 0; i < 2; i++)
                    {
                        ab[i] = new double[countIteration];
                        for (int j = 0; j < countIteration; j++)
#pragma warning disable CS8629 // Тип значения, допускающего NULL, может быть NULL.
                            ab[i][j] = (double) UniformDistribution.Generate(interval, generator);
#pragma warning restore CS8629 // Тип значения, допускающего NULL, может быть NULL.
                    }
                }


                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    Parallel.ForEach(rangePartitioner, range =>
                    {
                        Vectors x, matrixX, vectorE, y, calcTheta;
                        double stat;
                        Vectors paramDist = Vectors.InitVectors((1, 2));
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            double a = ab[0][i], b = ab[1][i];
                            if (a > b)
                                (a, b) = (b, a);
                            paramDist[0] = a; paramDist[1] = b;
                            x = LinespaceRandom.Generate((countObservations, model.CountFacts), [(a, b)], localGenerator.Value);
                            //x = UniformDistribution.Generate((countObservations, model.CountFacts), paramDist, localGenerator.Value);
                            matrixX = model.CreateMatrixX(x);
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = (matrixX & model.TrueTheta.T()).T() + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                            /*
                       stat = evolution is MMKEstimator ev
                           ? SREСriteriaLR.SRELR(ev.Config.Functions.LogLikelihoodFull, model, calcTheta, [paramsDist, x, y])
                           : SREСriteria.SRE(model, calcTheta, x, y);
                       */
                            stat = SREСriteria.SRE(model, calcTheta, x, y);

                            statistics[i] = stat;
                            if (debug)
                            {
                                lock (lockObj)
                                {
                                    //avg_stat += (stat - avg_stat) / (iter + 1);
                                    //avg_norm += (Vectors.Norm(model.TrueTheta - calcTheta) - avg_norm) / (iter + 1);

                                    Console.WriteLine(FormattableString.Invariant($"Итерация {iter}, S: {stat}, Theta: {calcTheta}"));

                                    // Обновление прогресс-бара требует синхронизации
                                    //progress.Add(iter, "Получение статистик SRE",
                                    //  FormattableString.Invariant($"Ср.Статистика: {avg_stat}, Ср.Норма: {avg_norm}"));
                                    //Console.ResetColor();
                                    iter++;
                                }
                            }
                        }

                    }
                    );
                }
                else
                {
                    double a, b;
                    for (int i = 0; i < countIteration; i++)

                    {

                        a = ab[0][i]; b = ab[1][i];
                        if (a > b)
                            (a, b) = (b, a);

                        x = LinespaceRandom.Generate((countObservations, model.CountFacts), [(a, b)], generator);

                        matrixX = model.CreateMatrixX(x);

#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.

                        vectorE = errorDist.Generate((1, countObservations), paramsDist);

#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.



#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        y = (matrixX & model.TrueTheta.T()).T() + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);

#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                        /*
                        stat = evolution is MMKEstimator ev
                            ? SREСriteriaLR.SRELR(ev.Config.Functions.LogLikelihoodFull, model, calcTheta, [paramsDist, x, y])
                            : SREСriteria.SRE(model, calcTheta, x, y);
                        */
                        stat = SREСriteria.SRE(model, calcTheta, x, y);
                        statistics[i] = stat;

                        if (debug)

                        {

                            avg_stat += (stat - avg_stat) / (i + 1);

                            avg_norm += (Vectors.Norm(model.TrueTheta - calcTheta) - avg_norm) / (i + 1);

                            progress.Add(i, "Получение статистик SRE", FormattableString.Invariant($"Ср.Статистика: {avg_stat}, Ср.Норма: {avg_norm}"));

                            Console.ResetColor();

                        }
                    }
                }
            }
            else 
            {
                ArgumentNullException.ThrowIfNull(planP);

                if (planX.Shape.Item2 != model.CountFacts) throw new ArgumentException("Число факторов в моделе не совпадает с размерностью точек плана");
                x = GenerateXFromPlan(planX, planP, countObservations, generator);
                Vectors u = (model.CreateMatrixX(x) & model.TrueTheta.T()).T();

                if (parallel)
                {
                    var rangePartitioner = Partitioner.Create(0, countIteration, countIteration / Environment.ProcessorCount);
                    Parallel.ForEach(rangePartitioner, range =>
                    {
                        Vectors vectorE, y, calcTheta;
                        double stat;
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                            vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            y = u + vectorE;
                            if (isRound && roundDecimals is not null)
                                y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                            calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);
                            /*
                       stat = evolution is MMKEstimator ev
                           ? SREСriteriaLR.SRELR(ev.Config.Functions.LogLikelihoodFull, model, calcTheta, [paramsDist, x, y])
                           : SREСriteria.SRE(model, calcTheta, x, y);
                       */
                            stat = SREСriteria.SRE(model, calcTheta, x, y);
                            statistics[i] = stat;
                            if (debug)
                            {
                                lock (lockObj)
                                {
                                    //avg_stat += (stat - avg_stat) / (iter + 1);
                                    //avg_norm += (Vectors.Norm(model.TrueTheta - calcTheta) - avg_norm) / (iter + 1);

                                    Console.WriteLine(FormattableString.Invariant($"Итерация {iter}, S: {stat}, Theta: {calcTheta}"));

                                    // Обновление прогресс-бара требует синхронизации
                                    //progress.Add(iter, "Получение статистик SRE",
                                    //  FormattableString.Invariant($"Ср.Статистика: {avg_stat}, Ср.Норма: {avg_norm}"));
                                    //Console.ResetColor();
                                    iter++;
                                }
                            }
                        }

                    }
                    );
                }
                else
                {
                    for (int i = 0; i < countIteration; i++)
                    {
#pragma warning disable CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
                        vectorE = errorDist.Generate((1, countObservations), paramsDist);
#pragma warning restore CS8600 // Преобразование литерала, допускающего значение NULL или возможного значения NULL в тип, не допускающий значение NULL.
#pragma warning disable CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.
                        y = u + vectorE;
                        if (isRound && roundDecimals is not null)
                            y = RoundVector(y, (int) roundDecimals);
#pragma warning restore CS8604 // Возможно, аргумент-ссылка, допускающий значение NULL.

                        calcTheta = evolution.EstimateParameters(model, [paramsDist, x, y]);

                        /*
                       stat = evolution is MMKEstimator ev
                           ? SREСriteriaLR.SRELR(ev.Config.Functions.LogLikelihoodFull, model, calcTheta, [paramsDist, x, y])
                           : SREСriteria.SRE(model, calcTheta, x, y);
                       */
                        stat = SREСriteria.SRE(model, calcTheta, x, y);

                        statistics[i] = stat;

                        if (debug)

                        {

                            avg_stat += (stat - avg_stat) / (i + 1);

                            avg_norm += (Vectors.Norm(model.TrueTheta - calcTheta) - avg_norm) / (i + 1);

                            progress.Add(i, "Получение статистик SRE", FormattableString.Invariant($"Ср.Статистика: {avg_stat}, Ср.Норма: {avg_norm}"));

                            Console.ResetColor();

                        }
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
}
