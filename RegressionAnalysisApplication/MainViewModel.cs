using System.Collections.ObjectModel;
using System.ComponentModel;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Windows;
using System.Windows.Input;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Win32;

using Microsoft.Extensions.Configuration;
using Newtonsoft.Json;
using RegressionAnalysisLibrary;

namespace RegressionAnalysisApplication
{

    public partial class MainWindowViewModel : ObservableObject
    {
        public ObservableCollection<Distribution> ErrorDistibution { get; set; } = [];
        public ObservableCollection<Regression> Regressions { get; set; } = [];

        public ObservableCollection<Evolution> MethodsEvolution { get; set; } = [];

        public ObservableCollection<SelectedParameter> SelectedParameters { get; set; } = [];

        [ObservableProperty]
        private bool _allParameter = false;

        [ObservableProperty]
        private Distribution? _selectDistribution;

        [ObservableProperty]
        private Regression? _selectRegression;

        [ObservableProperty]
        private Evolution? _selectEvolution;

        [ObservableProperty]
        private bool _roundedCheck = false;

        [ObservableProperty]
        private int _roundedValue = 1;

        [ObservableProperty]
        private int _dataMode = 0;

        [ObservableProperty]
        private int _valueProgress = 0;

        [ObservableProperty]
        private int _valueProgressParam = 0;

        [ObservableProperty]
        private string _textProgress = "";

        [ObservableProperty]
        private string _textProgressParam = "";

        [ObservableProperty]
        private bool _startButtonActive = true;

        [ObservableProperty]
        private bool _resetButtonActive = false;

        [ObservableProperty]
        private bool _initValGenCheck = false;

        [ObservableProperty]
        private int _initValGen = 324;

        [ObservableProperty]
        private bool _startButtonParamActive = true;

        [ObservableProperty]
        private bool _resetButtonParamActive = false;

        [ObservableProperty]
        private int _selectCriteria = 0;

        [ObservableProperty]
        private bool _parallelCheck = true;

        private int _countIteration = 2000;
        public int CountIteration {
            get => _countIteration;
            set
            {
                if (_countIteration != value && value > 0)
                {
                    _countIteration = value;
                    OnPropertyChanged(nameof(CountIteration));
                }
            }
        }

        [ObservableProperty]
        private string _fileTitle = "SRE MNK Uniform [0, 1] n500 N2000";

        [ObservableProperty]
        private string _fileData = "Не выбрано";

        private int _countObservations = 500;
        public int CountObservations {
            get => _countObservations;
            set {
                if (_countObservations != value) {
                    if (SelectRegression is not null)
                    {
                        if (SelectRegression.CountRegressor <= value)
                            _countObservations = value;
                    }
                    else if (value > 0) {
                        _countObservations = value;
                    }
                }
            }
        }

        [ObservableProperty]
        private string _savePath = "Не выбрано";

        [ObservableProperty]
        private int _currentTabIndex = 0;

        public IConfigurationRoot? Config;

        private CancellationTokenSource? _cancellationTokenSource;

        public ICommand ChoiseFilePath { get; }
        public ICommand OpenFilePath { get; }
        public ICommand UpdateCountObservations { get; }
        public ICommand ShowMessage { get; }
        public ICommand StartCalculation { get; }
        public ICommand StartCalculationParam { get; }
        public ICommand StopCalculation { get; }
        public ICommand StopCalculationParam { get; }
        public ICommand UpdateSelectParam { get; }

        public ICommand RefrashTitle { get; }

        public MainWindowViewModel() {
            LoadErrorDistribuions();
            LoadListRegressions();
            LoadEvolutions();
            if (ErrorDistibution.Count > 0)
                SelectDistribution = ErrorDistibution[0];
            if (Regressions.Count > 0) {
                SelectRegression = Regressions[0];
            }
            if (MethodsEvolution.Count > 0) {
                SelectEvolution = MethodsEvolution[0];
            }
            SelectedParameters = [new SelectedParameter("Параметр 0", true)];

            ChoiseFilePath = new RelayCommand(ChoiseFilePathFunction);
            OpenFilePath = new RelayCommand(OpenFilePathFunction);
            UpdateCountObservations = new RelayCommand<int>(UpdateCountObservationsFunction);
            ShowMessage = new RelayCommand<string>(ShowMessageBox);

            StartCalculation = new RelayCommand(StartCalculationFunction);
            StopCalculation = new RelayCommand(ResetCalculationFunction);

            StartCalculationParam = new RelayCommand(StartCalculationParamFunction);
            StopCalculationParam = new RelayCommand(ResetCalculationParamFunction);

            UpdateSelectParam = new RelayCommand<int>(SelectParamFunction);

            RefrashTitle = new RelayCommand(RefrashTitleFunction);

            if (File.Exists("config.ini"))
                Config = (new ConfigurationBuilder()).SetBasePath(Directory.GetCurrentDirectory()).AddIniFile("config.ini").Build();
        }

        private void RefrashTitleFunction() {
            
            if (SelectDistribution is not null && SelectEvolution is not null)
            {
                var sb = new StringBuilder();
                if (CurrentTabIndex < 0 || CurrentTabIndex > 1) return;
                switch (CurrentTabIndex)
                {
                    case 0:
                        sb.Append("SRE ");
                        break;
                    case 1:
                        sb.Append("P{k} ");
                        break;
                    default:
                        break;
                }
                sb.Append(SelectEvolution.ParameterEstimator.Name);
                sb.Append(' ');
                sb.Append(SelectDistribution.RandomDistribution.Name);
                sb.Append(" [");
                sb.Append($"{(int) SelectDistribution.Parameters[0].Value}");
                for (int i = 1; i < SelectDistribution.RandomDistribution.CountParametrsDistribution; i++)
                {
                    sb.Append($", {(int) SelectDistribution.Parameters[i].Value}");
                }
                sb.Append("] ");
                sb.Append($"n{CountObservations} N{CountIteration}");
                if (RoundedCheck) {
                    sb.Append($" R{RoundedValue}");
                }
                FileTitle = sb.ToString();
            }
            return;
        }

        private void SelectParamFunction(int newCount) {
            if (newCount > SelectedParameters.Count)
            {
                for (int i = SelectedParameters.Count; i < newCount; i++)
                {
                    SelectedParameters.Add(new SelectedParameter($"Параметр {i}", false));
                }
            }
            else if (newCount < SelectedParameters.Count) {
                for (int i = SelectedParameters.Count-1; i >= newCount; i--)
                {
                    SelectedParameters.RemoveAt(i);
                }
            }
        }
        private void UpdateCountObservationsFunction(int value) {
            _countObservations = value;
            OnPropertyChanged(nameof(CountObservations));
        } 

        private void ChoiseFilePathFunction() {
            var sb = new StringBuilder();
            string name = "document";
            if (SelectDistribution is not null && SelectEvolution is not null)
            {
                if (RoundedCheck) {
                    sb.Append(RoundedValue);
                    sb.Append('_');
                }
                sb.Append(SelectEvolution.ParameterEstimator.Name);
                sb.Append('_');
                sb.Append(SelectDistribution.RandomDistribution.Name);
                for (int i = 0; i < SelectDistribution.RandomDistribution.CountParametrsDistribution; i++) {
                    sb.Append($"_{(int) SelectDistribution.Parameters[i].Value}");
                }
                sb.Append($"_n{CountObservations}_N{CountIteration}");
                name = sb.ToString();
            }
            var path = SelectSaveFilePath(defaultFileName: name);
            if (path is not null)
                SavePath = path;
        }

        private void OpenFilePathFunction()
        {
            var path = OpenDataFilePath();
            if (path is not null)
                FileData = path;
        }

        private void ShowMessageBox(string? text) {
            if (text is not null)
                MessageBox.Show(text);
        }

        private void UpdateProgress(int percent)
        {
            ValueProgress = percent;
            TextProgress = $"{percent}%";
        }

        private void UpdateProgressParam(int percent)
        {
            ValueProgressParam = percent;
            TextProgressParam = $"{percent}%";
        }

        private async void StartCalculationFunction() {
            if (SelectEvolution is null) {
                ShowMessage.Execute("Ошибка: не выбран метод оценивания параметров регресии");
                return;
            }
            if (SelectDistribution is null)
            {
                ShowMessage.Execute($"Ошибка: не выбрано распределение ошибок");
                return;
            }

            var evolution = SelectEvolution.ParameterEstimator;
            if (evolution is MMPEstimator est)
            {
                est.Config = MMPConfigLoader.Load(SelectDistribution.Type);
            }
            if (evolution is MNKEstimator) {
                if (SelectCriteria == 1) {
                    ShowMessage.Execute($"Ошибка: LR-test может применятся только для ММП");
                    return;
                }
            }
            if (SelectRegression is null) {
                ShowMessage.Execute("Ошибка: не выбрана модель регрессии");
                return;
            }
            var model = SelectRegression.GetModel();
            string? directory = Path.GetDirectoryName(SavePath);
            if (!Directory.Exists(directory)) {
                ShowMessage.Execute("Ошибка: указанной папки сохранения не существует.");
                return;
            }
            if (!File.Exists(FileData) && DataMode != 0) {
                ShowMessage.Execute("Ошибка: файл загрузки не выбран");
                return;
            }


            if (DataMode == 0)
            {
                StartButtonActive = false;
                ResetButtonActive = true;
                StartButtonParamActive = false;
                ResetButtonParamActive = false;
                ValueProgress = 0;
                TextProgress = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgress);



                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }



                    var result = await Task.Run(() => RegressionEvaluator.Fit
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        typeCriteria: (Criteria) SelectCriteria,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token
                    ), _cancellationTokenSource.Token);
                    result.Statistics.SaveToDAT(path: SavePath, title: FileTitle);

                    TextProgress = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgress = "Вычисления отмененны";
                    ValueProgress = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }
            else if (DataMode == 1)
            {
                StartButtonActive = false;
                ResetButtonActive = true;
                StartButtonParamActive = false;
                ResetButtonParamActive = false;
                ValueProgress = 0;
                TextProgress = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgress);

                Vectors planX, weights;
                try
                {
                    var temp = ExperimentPlanLoader.LoadPlan(FileData);
                    planX = new Vectors(temp.points);
                    weights = new Vectors(temp.weights);
                }
                catch
                {
                    ShowMessage.Execute("Ошибка: файл загрузки не подходит под формат");
                    return;
                }

                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }

                    var result = await Task.Run(() => RegressionEvaluator.Fit
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        typeCriteria: (Criteria) SelectCriteria,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token,
                        planX: planX,
                        planP: weights
                    ), _cancellationTokenSource.Token);
                    result.Statistics.SaveToDAT(path: SavePath, title: FileTitle);

                    TextProgress = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgress = "Вычисления отмененны";
                    ValueProgress = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }
            else if (DataMode == 2) {
                StartButtonActive = false;
                ResetButtonActive = true;
                StartButtonParamActive = false;
                ResetButtonParamActive = false;
                ValueProgress = 0;
                TextProgress = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgress);

                Vectors observation;
                try
                {
                    observation = new Vectors(ObservationsLoader.LoadObservations(FileData));
                }
                catch
                {
                    ShowMessage.Execute("Ошибка: файл загрузки не подходит под формат");
                    return;
                }

                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }

                    var result = await Task.Run(() => RegressionEvaluator.Fit
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        typeCriteria: (Criteria) SelectCriteria,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token,
                        observations: observation
                    ), _cancellationTokenSource.Token);
                    result.Statistics.SaveToDAT(path: SavePath, title: FileTitle);

                    TextProgress = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgress = "Вычисления отмененны";
                    ValueProgress = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }

        }

        private async void StartCalculationParamFunction()
        {
            if (SelectEvolution is null)
            {
                ShowMessage.Execute("Ошибка: не выбран метод оценивания параметров регресии");
                return;
            }
            if (SelectDistribution is null)
            {
                ShowMessage.Execute($"Ошибка: не выбрано распределение ошибок");
                return;
            }

            var evolution = SelectEvolution.ParameterEstimator;
            if (evolution is MMPEstimator est)
            {
                est.Config = MMPConfigLoader.Load(SelectDistribution.Type, ismultiiteration: false);
            }
            if (SelectRegression is null)
            {
                ShowMessage.Execute("Ошибка не выбрана моедль регрессии");
                return;
            }
            var model = SelectRegression.GetModel();
            string? directory = Path.GetDirectoryName(SavePath);
            if (!Directory.Exists(directory))
            {
                ShowMessage.Execute("Ошибка: указанной папки сохранения не существует.");
                return;
            }
            if (!File.Exists(FileData) && DataMode != 0)
            {
                ShowMessage.Execute("Ошибка: файл загрузки не выбран");
                return;
            }

            List<int> selectedparam = [];
            if (AllParameter)
            {
                for (int i = 0; i < SelectedParameters.Count; i++)
                {
                    selectedparam.Add(i);
                }
            }
            else {
                for (int i = 0; i < SelectedParameters.Count; i++) {
                    if (SelectedParameters[i].Selected)
                        selectedparam.Add(i);
                }
            }

            if (selectedparam.Count == 0) {
                ShowMessage.Execute("Ошибка: не выбран не один параметр");
                return;
            }

            if (DataMode == 0)
            {
                StartButtonActive = false;
                ResetButtonActive = false;
                StartButtonParamActive = false;
                ResetButtonParamActive = true;
                ValueProgressParam = 0;
                TextProgressParam = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgressParam);



                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }



                    var result = await Task.Run(() => RegressionEvaluator.FitParameters
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        numberParametr: [.. selectedparam],
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token
                    ), _cancellationTokenSource.Token);
                    for (int i = 0; i < result.Statistics.Shape.Item1; i++) {
                        var item = Vectors.GetRow(result.Statistics, i);
                        var title = FileTitle;
                        if (FileTitle.Contains("{k}")) {
                            title = title.Replace("{k}", $"{i}");
                        }
                        item.SaveToDAT(path: SavePath.Replace(".dat", $"_{i}.dat"), title);
                    }

                    TextProgressParam = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgressParam = "Вычисления отмененны";
                    ValueProgressParam = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }
            else if (DataMode == 1)
            {
                StartButtonParamActive = false;
                ResetButtonParamActive = true;
                StartButtonActive = false;
                ResetButtonActive = false;
                ValueProgressParam = 0;
                TextProgressParam = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgressParam);

                Vectors planX, weights;
                try
                {
                    var temp = ExperimentPlanLoader.LoadPlan(FileData);
                    planX = new Vectors(temp.points);
                    weights = new Vectors(temp.weights);
                }
                catch
                {
                    ShowMessage.Execute("Ошибка: файл загрузки не подходит под формат");
                    return;
                }

                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }

                    var result = await Task.Run(() => RegressionEvaluator.FitParameters
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        numberParametr: [.. selectedparam],
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token,
                        planX: planX,
                        planP: weights
                    ), _cancellationTokenSource.Token);
                    
                    for (int i = 0; i < result.Statistics.Shape.Item1; i++)
                    {
                        var item = Vectors.GetRow(result.Statistics, i);
                        var title = FileTitle;
                        if (FileTitle.Contains("{k}"))
                        {
                            title = title.Replace("{k}", $"{i}");
                        }
                        item.SaveToDAT(path: SavePath.Replace(".dat", $"_{i}.dat"), title);
                    }
                    TextProgressParam = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgressParam = "Вычисления отмененны";
                    ValueProgressParam = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }
            else if (DataMode == 2)
            {
                StartButtonActive = false;
                ResetButtonActive = false;
                StartButtonParamActive = false;
                ResetButtonParamActive = true;
                ValueProgressParam = 0;
                TextProgressParam = "Выполнение...";

                _cancellationTokenSource = new CancellationTokenSource();
                var progress = new Progress<int>(UpdateProgressParam);

                Vectors observation;
                try
                {
                    observation = new Vectors(ObservationsLoader.LoadObservations(FileData));
                }
                catch
                {
                    ShowMessage.Execute("Ошибка: файл загрузки не подходит под формат");
                    return;
                }

                try
                {
                    (double, double)[] factorBound;
                    if (SelectRegression.SameBound)
                    {
                        var min = SelectRegression.SameFactorBound.Min;
                        var max = SelectRegression.SameFactorBound.Max;
                        if (min is null || max is null)
                            throw new ArgumentException("Границы введены не верно");
                        if (min > max)
                        {
                            factorBound = [((double) max, (double) min)];
                        }
                        factorBound = [((double) min, (double) max)];
                    }
                    else
                    {
                        factorBound = SelectRegression.ConvertBounds();
                    }

                    var result = await Task.Run(() => RegressionEvaluator.FitParameters
                    (
                        model: model,
                        evolution: evolution,
                        countIteration: CountIteration,
                        countObservations: CountObservations,
                        numberParametr: [.. selectedparam],
                        errorDist: SelectDistribution.RandomDistribution,
                        paramsDist: SelectDistribution.ConvertParamToVector(),
                        parallel: ParallelCheck,
                        isRound: RoundedCheck,
                        roundDecimals: RoundedValue,
                        dimension: factorBound,
                        progress: progress,
                        seed: InitValGenCheck ? InitValGen : null,
                        token: _cancellationTokenSource.Token,
                        observations: observation
                    ), _cancellationTokenSource.Token);
                    
                    for (int i = 0; i < result.Statistics.Shape.Item1; i++)
                    {
                        var item = Vectors.GetRow(result.Statistics, i);
                        var title = FileTitle;
                        if (FileTitle.Contains("{k}"))
                        {
                            title = title.Replace("{k}", $"{i}");
                        }
                        item.SaveToDAT(path: SavePath.Replace(".dat", $"_{i}.dat"), title);
                    }
                    TextProgressParam = "Вычисление завершины!";
                }
                catch (OperationCanceledException)
                {
                    TextProgressParam = "Вычисления отмененны";
                    ValueProgressParam = 0;
                }
                catch (Exception ex)
                {
                    TextProgress = "Ошибка";
                    ShowMessageBox($"Ошибка: {ex.Message}");
                }
                finally
                {
                    _cancellationTokenSource.Dispose();
                    _cancellationTokenSource = null;
                    StartButtonActive = true;
                    ResetButtonActive = false;
                    StartButtonParamActive = true;
                    ResetButtonParamActive = false;
                }
            }

        }

        private void ResetCalculationFunction()
        {
            _cancellationTokenSource?.Cancel();
            ResetButtonActive = false;
            StartButtonActive = true;
        }

        private void ResetCalculationParamFunction()
        {
            _cancellationTokenSource?.Cancel();
            ResetButtonParamActive = false;
            StartButtonParamActive = true;
        }

        private string? OpenDataFilePath(string defaultFileName = "document",
                               string defaultExt = ".json",
                               string filter = "Файлы данных (*.json)|*.json")
        {
            var dialog = new OpenFileDialog
            {
                FileName = defaultFileName,    // Имя файла по умолчанию
                DefaultExt = defaultExt,       // Расширение по умолчанию
                Filter = filter,               // Фильтры файлов
                InitialDirectory = Config is null ? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
                                                  : Config["AppSettings:BaseFolder"],
                Title = "Выберите файл для загрузки",
                AddExtension = true
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        private string? SelectSaveFilePath(string defaultFileName = "document",
                               string defaultExt = ".dat",
                               string filter = "Файлы данных (*.dat)|*.dat")
        {
            var dialog = new SaveFileDialog
            {
                FileName = defaultFileName,    // Имя файла по умолчанию
                DefaultExt = defaultExt,       // Расширение по умолчанию
                Filter = filter,               // Фильтры файлов
                InitialDirectory = Config is null ? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
                                                  : Config["AppSettings:BaseFolder"],
                Title = "Выберите место для сохранения файла",
                AddExtension = true,
                OverwritePrompt = true         // Предупреждать о перезаписи
            };

            return dialog.ShowDialog() == true ? dialog.FileName : null;
        }

        private void LoadListRegressions() {
            var result = new List<Regression>();
            var fabric = new RegressionFactory();
            foreach (var (name, instance) in fabric.GetAllRegressions())
            {
                result.Add(new Regression(this, name, instance));
            }
            Regressions = new ObservableCollection<Regression>(result);
        }

        private void LoadEvolutions()
        {
            var result = new List<Evolution>();
            var fabric = new EvolutionFactory();
            foreach (var (name, instance) in fabric.GetAllEvolutions())
            {
                result.Add(new Evolution(name, instance));
            }
            MethodsEvolution = new ObservableCollection<Evolution>(result);
        }

        private void LoadErrorDistribuions() {
            var result = new List<Distribution>();
            var fabric = new DistributionFactory();
            Vectors defaultParam;
            string[] nameParam;
            (double?, double?)[]? bounds;
            FactorBound? factorBound;
            foreach (var elem in fabric.GetAllDistributions()) {
                var parameters = new List<Parameter>();
                defaultParam = elem.Instance.DefaultParametrs;
                nameParam = elem.Instance.NameParameters;
                bounds = elem.Instance.BoundsParameters;
                if (bounds is not null) {
                    for (int i = 0; i < elem.Instance.CountParametrsDistribution; i++) {
                        factorBound = new FactorBound(bounds[i].Item1, bounds[i].Item2);
                        parameters.Add(new Parameter(nameParam[i], defaultParam[i], factorBound));
                    }
                }
                else {
                    factorBound = null;
                    for (int i = 0; i < elem.Instance.CountParametrsDistribution; i++)
                    {
                        parameters.Add(new Parameter(nameParam[i], defaultParam[i], factorBound));
                    }
                }
                result.Add(new Distribution(elem.Name, elem.Type, parameters, elem.Instance));
            }
            ErrorDistibution = new ObservableCollection<Distribution>(result);
        }
    }
    public class ViewModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        protected bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(field, value))
                return false;

            field = value;
            OnPropertyChanged(propertyName);
            return true;
        }
    }


    public partial class Evolution : ObservableObject {
        [ObservableProperty]
        private string _name;

        public IParameterEstimator ParameterEstimator { get; set; }

        public Evolution(string name, IParameterEstimator parameterEstimator) {
            Name = name;
            ParameterEstimator = parameterEstimator;
        }
    }

    public partial class Regression : ObservableObject {

        [ObservableProperty]
        private MainWindowViewModel _main;

        [ObservableProperty]
        private string _name;

        private bool _freeMember = false;

        public bool FreeMember {
            get => _freeMember;
            set {
                if (value != _freeMember) {
                    if (value)
                    {
                        TrueTheta.Insert(0, new Parameter("Параметр 0", 1.0, null));
                    }
                    else {
                        TrueTheta.RemoveAt(0);
                    }
                    _freeMember = value;
                    OnPropertyChanged(nameof(FreeMember));
                    OnPropertyChanged(nameof(CountRegressor));
                    UpdateNameThetaParameters();
                }
            }
        }

        private int _countFacts = 1;

        public int CountFacts {
            get => _countFacts;
            set {
                if (_countFacts != value && value >= 0) {
                    UpdateFacts(value);
                    _countFacts = value;
                    OnPropertyChanged(nameof(CountFacts));
                    OnPropertyChanged(nameof(CountRegressor));
                }
            }
        }

        public int CountRegressor
        {
            get
            {
                var countfacts = 0;
                for (int i = 0; i < Factors.Count; i++)
                    if (Factors[i].Active)
                        countfacts++;
                var result = countfacts + SelectAdditionalFactors.Count + (FreeMember ? 1 : 0);
                if (Main.CountObservations < result)
                {
                    Main.UpdateCountObservations.Execute(result);
                }
                Main.UpdateSelectParam.Execute(result);
                return result;
            }
        }

        public ObservableCollection<Parameter> TrueTheta { get; set; }
        public ObservableCollection<AdditionalFactor> AdditionalFactors { get; set; }
        public ObservableCollection<Factor> Factors { get; set; }
        public HashSet<AdditionalFactor> SelectAdditionalFactors { get; set; }
        public List<int> CheckedFactors { get; set; }

        public ObservableCollection<FactorBound> FactorBounds { get; set; }

        [ObservableProperty]
        private FactorBound _sameFactorBound = new(-1, 1);

        [ObservableProperty]
        private bool _sameBound = true;

        private bool _selectedParam = false;

        public IModel ModelRegression { get; set; }
        public ICommand ResetFactor { get; }
        public ICommand ResetAdditionalRegressors { get; }
        public ICommand SelectAllFactors { get; }
        public ICommand SelectAllAdditionalRegressor { get; }
        public ICommand CheckFactor { get; }
        public ICommand UnCheckFactor { get; }

        public ICommand UpdateCountRegressor { get; }
        public ICommand CheckAdditionalRegressor { get; }
        public ICommand UnCheckAdditionalRegressor { get; }

        public Regression(MainWindowViewModel main, string name, IModel model) {
            Name = name;
            Main = main;
            ModelRegression = model;
            Factors = [new Factor(this, 0)];
            AdditionalFactors = [];
            FreeMember = false;
            TrueTheta = [ new Parameter("Параметр 0", 1.0, null) ];
            SelectAdditionalFactors = [];
            CheckedFactors = [];

            CheckFactor = new RelayCommand<int>(CheckFactorFunction);
            UnCheckFactor = new RelayCommand<int>(UnCheckFactorFunction);

            ResetFactor = new RelayCommand(ResetFactorFunction);
            ResetAdditionalRegressors = new RelayCommand(ResetAdditionalRegressorsFunction);

            SelectAllFactors = new RelayCommand(SelectAllFactorsFunction);
            SelectAllAdditionalRegressor = new RelayCommand(SelectAllAdditionalRegressorsFunction);

            CheckAdditionalRegressor = new RelayCommand<AdditionalFactor>(CheckAdditionalRegressorFunction);
            UnCheckAdditionalRegressor = new RelayCommand<AdditionalFactor>(UnCheckAdditionalRegressorFunction);

            UpdateCountRegressor = new RelayCommand(() => OnPropertyChanged(nameof(CountRegressor)));
            FactorBounds = [new FactorBound(-1, 1)];
        }

        public IModel GetModel() {
            UpdateModel();
            return ModelRegression;
        }

        public (double, double)[] ConvertBounds() {
            return FactorBounds.Select(x => {
                var min = x.Min; var max = x.Max;
                if (min is null || max is null)
                    throw new ArgumentException("Ошибка: Границы факторов построенные не верно");
                if (min > max)
                    (min, max) = (max, min);
                return ((double) min, (double) max); 
            }).ToArray();
        }

        public void UpdateModel()
        {
            var additionalRegressors = SelectAdditionalFactors.OrderBy(x => x.Val.Compare()).ThenBy(x => x.Name).Select(x => x.Val).ToList();
            List<int> notActiveFactor = [];
            for (int i = 0; i < CountFacts; i++) {
                if (!Factors[i].Active)
                    notActiveFactor.Add(i);
            }
            ModelRegression.Update(CountFacts, additionalRegressors, notActiveFactor, new Vectors(TrueTheta.Select(x => x.Value).ToArray()), FreeMember);
        }

        private void CheckAdditionalRegressorFunction(AdditionalFactor? factor) {
            if (factor is not null)
            {
                SelectAdditionalFactors.Add(factor);
                TrueTheta.Add(new Parameter($"Параметр {TrueTheta.Count}", 1.0, null));
                OnPropertyChanged(nameof(CountRegressor));
            }
        }

        private void UnCheckAdditionalRegressorFunction(AdditionalFactor? factor) {
            if (factor is not null)
            {
                SelectAdditionalFactors.Remove(factor);
                TrueTheta.RemoveAt(TrueTheta.Count - 1);
                OnPropertyChanged(nameof(CountRegressor));
            }
        }

        private void SelectAllAdditionalRegressorsFunction() {
            AdditionalFactors.Clear();
            if (_selectedParam)
            {
                var additionalfactors = ModelRegression.GenerateAdditionalRegressors([.. CheckedFactors], 10);
                foreach (var factor in additionalfactors)
                {
                    var elem = new AdditionalFactor(this, factor);
                    if (SelectAdditionalFactors.TryGetValue(elem, out var temp))
                        AdditionalFactors.Add(temp);
                    else
                        AdditionalFactors.Add(new AdditionalFactor(this, factor));
                }
                _selectedParam = false;
            }
            else {
                var sortedlist = SelectAdditionalFactors.ToList().OrderBy(x => x.Val.Compare()).ThenBy(x => x.Name);
                foreach (var factor in sortedlist) {
                    AdditionalFactors.Add(factor);
                }
                _selectedParam = true;
            }
        }

        private void SelectAllFactorsFunction() {
            CheckedFactors.Clear();
            foreach (var factor in Factors) {
                factor.Active = true;
            }
            OnPropertyChanged(nameof(CountRegressor));
        }

        private void ResetFactorFunction() {
            CheckedFactors.Clear();
            AdditionalFactors.Clear();
            foreach (var factor in Factors) {
                factor.Reset();
            }
            OnPropertyChanged(nameof(CountRegressor));
        }

        private void ResetAdditionalRegressorsFunction() {
            foreach (var factor in AdditionalFactors) {
                factor.Reset();
            }
            SelectAdditionalFactors.Clear();
            OnPropertyChanged(nameof(CountRegressor));
        }

        private void CheckFactorFunction(int numfactor) {
            if (numfactor < 0) return;
            CheckedFactors.Add(numfactor);
            AdditionalFactors.Clear();
            var additionalfactors = ModelRegression.GenerateAdditionalRegressors([.. CheckedFactors], 10);
            foreach (var factor in additionalfactors)
            {
                var elem = new AdditionalFactor(this, factor);
                if (SelectAdditionalFactors.TryGetValue(elem, out var temp))
                    AdditionalFactors.Add(temp);
                else
                    AdditionalFactors.Add(elem);
            }
        }

        private void UnCheckFactorFunction(int numfactor)
        {
            if (numfactor < 0) return;
            CheckedFactors.Remove(numfactor);
            var additionalfactors = ModelRegression.GenerateAdditionalRegressors([.. CheckedFactors], 10);
            AdditionalFactors.Clear();
            foreach (var factor in additionalfactors)
            {
                var elem = new AdditionalFactor(this, factor);
                if (SelectAdditionalFactors.TryGetValue(elem, out var temp))
                    AdditionalFactors.Add(temp);
                else
                    AdditionalFactors.Add(new AdditionalFactor(this, factor));
            }
        }

        private void UpdateFacts(int newsize) {
            if (newsize > Factors.Count)
            {
                for (int i = Factors.Count; i < newsize; i++)
                {
                    Factors.Add(new Factor(this, i));
                    FactorBounds.Add(new FactorBound(-1, 1));
                    TrueTheta.Insert(i + (FreeMember ? 1: 0), new Parameter($"Параметр {i + (FreeMember ? 1 : 0)}", 1.0, null));
                }
                UpdateNameThetaParameters();
            }
            else if (newsize < Factors.Count) {
                for (int i = Factors.Count - 1; i >= newsize; i--)
                {
                    Factors.RemoveAt(i);
                    FactorBounds.RemoveAt(i);
                    TrueTheta.RemoveAt(i + (FreeMember ? 1 : 0));
                }
                AdditionalFactors.Clear();
                SelectAdditionalFactors.Clear();
            }
        }
        public void UpdateNameThetaParameters()
        {
            for (int i = 0; i < TrueTheta.Count; i++) {
                TrueTheta[i].Name = $"Параметр {i}";
            }
        }
    }
    

    public partial class Factor : ObservableObject {

        [ObservableProperty]
        private Regression _main;
        
        [ObservableProperty]
        private string _name;

        [ObservableProperty]
        private int _val;

        private bool _active = true;

        public bool Active 
        {
            get => _active;
            set {
                _active = value;
                OnPropertyChanged(nameof(Active));
                if (value)
                    Main.TrueTheta.Insert(Val + (Main.FreeMember ? 1 : 0), new Parameter($"Параметр {Val + (Main.FreeMember ? 1 : 0)}", 1.0, null));
                else
                    Main.TrueTheta.RemoveAt(Val + (Main.FreeMember ? 1 : 0));
                Main.UpdateNameThetaParameters();
                Main.UpdateCountRegressor.Execute(null);
            }
        }

        private bool _check = false;

        public bool Check {
            get => _check;
            set {
                if (value != _check) {
                    if (value)
                        Main.CheckFactor.Execute(Val);
                    else
                        Main.UnCheckFactor.Execute(Val);
                    _check = value;
                    OnPropertyChanged(nameof(Check));
                }
            }
        }

        public Factor(Regression main, int value) {
            Main = main;
            Val = value;
            Name = $"Фактор {value}";
        }
        public Factor(Regression main, int value, string name)
        {
            Main = main;
            Val = value;
            Name = name;
        }

        public void Reset() {
            Active = true;
            _check = false;
            OnPropertyChanged(nameof(Check));
        }
        public override bool Equals(object? obj)
        {
            return obj is Factor other && Name == other.Name;
        }

        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }

    }

    public partial class AdditionalFactor : ObservableObject
    {

        [ObservableProperty]
        private Regression _main;
        
        [ObservableProperty]
        private string _name;

        [ObservableProperty]
        private IAdditionalRegressor _val;

        private bool _check = false;

        public bool Check {
            get => _check;
            set {
                if (_check != value) {
                    _check = value;
                    OnPropertyChanged(nameof(Check));
                    if (value)
                        Main.CheckAdditionalRegressor.Execute(this);
                    else
                        Main.UnCheckAdditionalRegressor.Execute(this);
                }
            }
        }

        public AdditionalFactor(Regression main, IAdditionalRegressor value)
        {
            Main = main;
            Val = value;
            Name = value.ToString() ?? "";
        }

        public void Reset() {
            _check = false;
            OnPropertyChanged(nameof(Check));
        }

        public override bool Equals(object? obj) {
            return obj is AdditionalFactor other && Name == other.Name;
        }

        public override int GetHashCode()
        {
            return Name.GetHashCode();
        }
    }

    public class Distribution(string name, TypeDisribution type, IEnumerable<Parameter> parameters, IRandomDistribution randomDistribution) : ObservableObject
    {
        public string Name { get; } = name;
        public IRandomDistribution RandomDistribution { get; } = randomDistribution;
        public ObservableCollection<Parameter> Parameters { get; } = new ObservableCollection<Parameter>(parameters);

        public TypeDisribution Type { get; } = type;

        public Vectors ConvertParamToVector() {
            return new Vectors(Parameters.Select(x => x.Value).ToArray());
        }
    }
    public partial class Parameter : ObservableObject
    {
        [ObservableProperty]
        private string _name;

        private double _value;

        public double Value 
        { 
            get => _value; 
            set {
                if (Bounds is not null)
                {
                    if (Bounds.Min is not null && value < Bounds.Min)
                        return;
                    else if (Bounds.Max is not null && value > Bounds.Max)
                        return;
                    else
                        _value = value;
                }
                else {
                    _value = value;
                }
                OnPropertyChanged(nameof(Value));
            } 
        }

        public FactorBound? Bounds { get; }

        public Parameter(string name, double value, FactorBound? bound)
        {
            Name = name;
            Value = value;
            Bounds = bound;
        }
    }

    public partial class SelectedParameter : ObservableObject {
        [ObservableProperty]
        private string _name;

        [ObservableProperty]
        private bool _selected;
        public SelectedParameter(string name, bool select) {
            Name = name;
            Selected = select;
        }
    }

    public partial class FactorBound(double? min, double? max) : ObservableObject
    {
        [ObservableProperty]
        private double? _min = min;

        [ObservableProperty]
        private double? _max = max;
    }

    public static class ObservationsLoader
    {
        public static double[][] LoadObservations(string jsonPath)
        {
            try
            {
                string json = File.ReadAllText(jsonPath);
                var observations = JsonConvert.DeserializeObject<double[][]>(json);

                return observations == null || observations.Length == 0 || observations[0].Length == 0
                    ? throw new InvalidDataException("Некорректные данные выборки")
                    : observations;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Ошибка загрузки выборки: {ex.Message}", ex);
            }
        }
    }


    public static class ExperimentPlanLoader
    {
        public class PlanPoint
        {
            public required double[] Point { get; set; }  // Координаты точки
            public required double Weight { get; set; }   // Вес точки
        }

        public static (double[][] points, double[] weights) LoadPlan(string jsonPath)
        {
            try
            {
                string json = File.ReadAllText(jsonPath);
                var plan = JsonConvert.DeserializeObject<PlanPoint[]>(json);

                if (plan == null || plan.Length == 0)
                    throw new InvalidDataException("Некорректные данные плана");

                // Проверка весов
                double totalWeight = plan.Sum(p => p.Weight);
                if (totalWeight > 1.0 + double.Epsilon)
                    throw new InvalidDataException($"Сумма весов ({totalWeight}) превышает 1");

                // Разделяем точки и веса
                double[][] points = plan.Select(p => p.Point).ToArray();
                double[] weights = plan.Select(p => p.Weight).ToArray();

                return (points, weights);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Ошибка загрузки плана: {ex.Message}", ex);
            }
        }
    }
}
