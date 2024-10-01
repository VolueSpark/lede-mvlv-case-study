
class Forecast:
    def __init__(self, uuid):
        self.uuid = uuid


    def predict(self):

        model_class = config['ml']['variant']
        library_path = f"lib.ml.model.{model_class.lower()}"
        Network = __import__(library_path, fromlist=[model_class])

        params={
            'window':
                {
                    'input_width': 48,
                    'label_width': 24,
                },
            'batch_size': 1,
            'shuffle': False
        }

        scaler = joblib.load(os.path.join(self.path, 'minmax_scaler.pkl'))
        test_loader = DataLoader(
            pl.read_parquet(os.path.join(self.path, 'test_scaled.parquet')),
            params=params,
            name='test'
        )

        model = getattr(Network, model_class)(
            work_dir=self.path,
            inputs_shape=test_loader.inputs_shape,
            inputs_exo_shape=test_loader.inputs_exo_shape,
            targets_shape = test_loader.targets_shape)
        model.load_state_dict(torch.load(os.path.join(self.path, 'model.pytorch'), weights_only=True))

        model.eval()

        with torch.no_grad():
            for i, data in enumerate((pbar := tqdm(test_loader, position=0, leave=True))):
                (_, inputs, inputs_exo, targets) = data.values()

                outputs = model(inputs, inputs_exo)