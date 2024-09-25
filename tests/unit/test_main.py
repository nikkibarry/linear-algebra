from linear_algebra.main import main


class TestMain:
    def test_main(self):
        try:
            main()
            assert False
        except NotImplementedError:
            assert True
