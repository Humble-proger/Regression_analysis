using System.Reflection;
using System.Text;

namespace RegressionAnalysisLibrary
{
    public class RegressionFactory
    {
        private readonly Dictionary<string, IModel> _regressions;

        public RegressionFactory()
        {
            _regressions = [];
            LoadRegression();
        }

        private void LoadRegression()
        {
            var assembly = Assembly.GetExecutingAssembly();

            foreach (var type in assembly.GetTypes()
                .Where(t => typeof(IModel).IsAssignableFrom(t)
                        && !t.IsInterface
                        && !t.IsAbstract))
            {
                var attr = type.GetCustomAttribute<RegressionAttribute>();
                if (attr == null) continue;

                try
                {
                    var instance = Activator.CreateInstance(type) as IModel;
#pragma warning disable CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                    _regressions[attr.Name] = instance;
#pragma warning restore CS8601 // Возможно, назначение-ссылка, допускающее значение NULL.
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to create {attr.Name}: {ex.Message}");
                }
            }
        }

        public IModel GetRegression(string type)
        {
            return _regressions.TryGetValue(type, out var regression)
                ? regression
                : throw new KeyNotFoundException($"Regression {type} not found");
        }

        public IEnumerable<(string Name, IModel Instance)> GetAllRegressions()
        {
            return _regressions.Select(kv => (kv.Key, kv.Value));
        }
    }

    public interface IModel
    {
        public bool FreeMember { get; set; }
        public int CountFacts { get; set; }
        public int CountRegressor { get; set; }
        public Vectors TrueTheta { get; set; }
        public List<IAdditionalRegressor> AdditionalFacts { get; set; }
        public List<int> NotActiveFacts { get; set; }

        public Vectors VectorFunc(Vectors x);
        public double True_value(Vectors x);
        public Vectors CreateMatrixX(Vectors x);

        public void Update(int count_facts, IEnumerable<IAdditionalRegressor> additional_facts, IEnumerable<int> nonActiveFacts, Vectors truetheta, bool free_member = false);
        public List<IAdditionalRegressor> GenerateAdditionalRegressors(int[] arr, int limit);
    }


    [AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
    public class RegressionAttribute(string name) : Attribute
    {
        public string Name { get; } = name;
    }

    public interface IAdditionalRegressor
    {
        public double Calc(Vectors valuesX);
        public double Compare();
    }


    public class RelatedFact : IAdditionalRegressor
    {
        public List<int> ListRelatedFacts { get; set; }

        public RelatedFact(IEnumerable<int> list)
        {
            ListRelatedFacts = new List<int>(list);
            ListRelatedFacts = [.. ListRelatedFacts.Order()];
        }

        public double Calc(Vectors valuesX)
        {
            var res = 1.0;
            try
            {
                foreach (var i in ListRelatedFacts)
                    res *= valuesX[i];
            }
            catch (Exception ex)
            {
                throw new Exception($"Ошибка при подсчёте взаимосвязных факторов - {ex}");
            }
            return res;
        }

        public double Compare()
        {
            return ListRelatedFacts.Count;
        }

        public override string ToString()
        {
            if (ListRelatedFacts.Count > 0)
            {
                var b = new StringBuilder($"Фактор {ListRelatedFacts[0]}");
                for (var i = 1; i < ListRelatedFacts.Count; i++)
                    b.Append($" x Фактор {ListRelatedFacts[i]}");
                return b.ToString();
            }
            return "";
        }
    }

    [Regression("Линейная регрессия")]
    public class LiniarModel : IModel
    {
        public bool FreeMember { get; set; } = false;
        public int CountFacts { get; set; } = 1;
        public int CountRegressor { get; set; } = 1;
        public Vectors TrueTheta { get; set; }
        public List<IAdditionalRegressor> AdditionalFacts { get; set; }

        public List<int> NotActiveFacts { get; set; }

        public LiniarModel()
        {
            TrueTheta = new Vectors([[]]);
            AdditionalFacts = [];
            NotActiveFacts = [];
        }

        public LiniarModel(int count_facts, IEnumerable<IAdditionalRegressor> related_facts, IEnumerable<int> nonActiveFacts, Vectors truetheta, bool free_member = false)
        {
            CountFacts = count_facts;
            FreeMember = free_member;
            // Фильтр для связанных фактов

            AdditionalFacts = new List<IAdditionalRegressor>(related_facts);
            AdditionalFacts = [.. AdditionalFacts.OrderBy(x => x.Compare())];

            NotActiveFacts = new List<int>(nonActiveFacts);
            if (NotActiveFacts.Count > CountFacts)
                throw new ArgumentException("List неактивных параметров передан не верно");
            foreach (var elem in NotActiveFacts)
                if (elem < 0 || elem >= CountFacts)
                    throw new ArgumentException("List неактивных параметров передан не верно");
            // Считаем количество регрессор
            CountRegressor = count_facts - NotActiveFacts.Count;
            if (free_member)
                CountRegressor += 1;
            CountRegressor += AdditionalFacts.Count;
            if (truetheta.Shape.Item2 == 1)
                truetheta = truetheta.T();
            else if (truetheta.Shape.Item1 != 1)
                throw new Exception("Vector truetheta cannot be a matrix.");
            if (truetheta.Size != CountRegressor)
                throw new Exception("Incorrect input of true theta values.");
            TrueTheta = truetheta;
        }

        public void Update(int count_facts, IEnumerable<IAdditionalRegressor> related_facts, IEnumerable<int> nonActiveFacts, Vectors truetheta, bool free_member = false)
        {
            CountFacts = count_facts;
            FreeMember = free_member;
            // Фильтр для связанных фактов

            AdditionalFacts = new List<IAdditionalRegressor>(related_facts);
            AdditionalFacts = [.. AdditionalFacts.OrderBy(x => x.Compare())];

            NotActiveFacts = new List<int>(nonActiveFacts);
            if (NotActiveFacts.Count > CountFacts)
                throw new ArgumentException("List неактивных параметров передан не верно");
            foreach (var elem in NotActiveFacts)
                if (elem < 0 || elem >= CountFacts)
                    throw new ArgumentException("List неактивных параметров передан не верно");
            // Считаем количество регрессор
            CountRegressor = count_facts - NotActiveFacts.Count;
            if (free_member)
                CountRegressor += 1;
            CountRegressor += AdditionalFacts.Count;
            if (truetheta.Shape.Item2 == 1)
                truetheta = truetheta.T();
            else if (truetheta.Shape.Item1 != 1)
                throw new Exception("Vector truetheta cannot be a matrix.");
            if (truetheta.Size != CountRegressor)
                throw new Exception("Incorrect input of true theta values.");
            TrueTheta = truetheta;
        }
        public Vectors VectorFunc(Vectors x)
        {
            if (x.Shape.Item2 == 1)
                x = x.T();
            else if (x.Shape.Item1 != 1)
                throw new Exception("Vector x cannot be a matrix.");
            if (x.Size != CountFacts) throw new Exception($"The size of the vector x does not match CountFacts. Number of facts: {CountFacts}. Size of x: {x.Size}.");
            var result = new double[CountRegressor];
            var index = 0;
            if (FreeMember)
                result[index++] = 1;
            for (var i = 0; i < CountFacts; i++)
                if (!NotActiveFacts.Contains(i))
                    result[index++] = x[0, i];
            for (var i = 0; i < AdditionalFacts.Count; i++, index++)
                result[index] = AdditionalFacts[i].Calc(x);
            return new Vectors(result);
        }
        public double True_value(Vectors x)
        {
            if (x.Shape.Item2 == 1)
                x = x.T();
            else if (x.Shape.Item1 != 1)
                throw new Exception("Vector x cannot be a matrix.");
            if (x.Size != CountFacts) throw new Exception($"The size of the vector x does not match CountFacts. Number of facts: {CountFacts}. Size of x: {x.Size}.");
            var index = 0;
            double result = 0;
            if (FreeMember)
                result += TrueTheta[0, index++];
            for (var i = 0; i < CountFacts; i++)
                if (!NotActiveFacts.Contains(i))
                    result += x[0, i] * TrueTheta[0, index++];
            for (var i = 0; i < AdditionalFacts.Count; i++, index++)
                result += AdditionalFacts[i].Calc(x) * TrueTheta[0, index];
            return result;
        }
        public Vectors CreateMatrixX(Vectors x)
        {
            if (x.Shape.Item2 != CountFacts)
                throw new Exception($"Incorrect matrix size x.The matrix row size is {x.Shape.Item2}, but should be {CountFacts}");
            var result = Vectors.InitVectors((x.Shape.Item1, CountRegressor));
            for (var i = 0; i < x.Shape.Item1; i++)
                result.SetRow(VectorFunc(Vectors.GetRow(x, i)), i);
            return result;
        }

        public List<IAdditionalRegressor> GenerateAdditionalRegressors(int[] arr, int limit)
        {
            if (arr.Length <= 1)
                return [];
            Array.Sort(arr); // Сортируем массив по возрастанию
            var result = new List<IAdditionalRegressor>();

            // Начинаем с комбинаций максимальной длины (размер массива), затем уменьшаем
            for (var r = arr.Length; r >= 2; r--)
            {
                // Генерируем все комбинации длины r
                var combinations = GetCombinations(arr, r);
                result.AddRange(combinations);

                // Если достигли лимита, останавливаемся
                if (result.Count >= limit)
                    break;
            }

            // Возвращаем только первые `limit` элементов
            return result.Take(limit).ToList();
        }

        static List<RelatedFact> GetCombinations(int[] arr, int k)
        {
            var result = new List<RelatedFact>();
            var n = arr.Length;

            // Рекурсивный метод для генерации комбинаций
            void Generate(int start, List<int> current)
            {
                if (current.Count == k)
                {
                    result.Add(new RelatedFact(current));
                    return;
                }

                for (var i = start; i < n; i++)
                {
                    current.Add(arr[i]);
                    Generate(i + 1, current);
                    current.RemoveAt(current.Count - 1);
                }
            }

            Generate(0, []);
            return result;
        }
    }
}