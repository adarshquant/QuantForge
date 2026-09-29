from pathlib import Path


NIFTY_50 = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "MAXHEALTH.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS"
]


NIFTY_NEXT_50 = [
    "ABB.NS",
    "ACC.NS",
    "ADANIGREEN.NS",
    "ADANIPOWER.NS",
    "AMBUJACEM.NS",
    "BANKBARODA.NS",
    "BANKINDIA.NS",
    "BOSCHLTD.NS",
    "CANBK.NS",
    "CHOLAFIN.NS",
    "COLPAL.NS",
    "DABUR.NS",
    "DIVISLAB.NS",
    "DLF.NS",
    "DMART.NS",
    "GAIL.NS",
    "GODREJCP.NS",
    "GODREJPROP.NS",
    "HAL.NS",
    "HAVELLS.NS",
    "ICICIGI.NS",
    "ICICIPRULI.NS",
    "IDFCFIRSTB.NS",
    "INDHOTEL.NS",
    "INDIGO.NS",
    "IOC.NS",
    "IRCTC.NS",
    "JINDALSTEL.NS",
    "JSWENERGY.NS",
    "LICI.NS",
    "LODHA.NS",
    "LUPIN.NS",
    "MARICO.NS",
    "MOTHERSON.NS",
    "MPHASIS.NS",
    "MUTHOOTFIN.NS",
    "NAUKRI.NS",
    "NHPC.NS",
    "OFSS.NS",
    "PAYTM.NS",
    "PERSISTENT.NS",
    "PIDILITIND.NS",
    "PFC.NS",
    "PNB.NS",
    "POLYCAB.NS",
    "RECLTD.NS",
    "SIEMENS.NS",
    "SRF.NS",
    "TORNTPHARM.NS",
    "TVSMOTOR.NS"
]


EXTENDED_50 = [
    "AARTIIND.NS",
    "ABCAPITAL.NS",
    "ABFRL.NS",
    "ALKEM.NS",
    "AUROPHARMA.NS",
    "BALKRISIND.NS",
    "BANDHANBNK.NS",
    "BHARATFORG.NS",
    "BIOCON.NS",
    "BHEL.NS",
    "BLUESTARCO.NS",
    "BRITANNIA.NS",
    "CGPOWER.NS",
    "CUMMINSIND.NS",
    "DALBHARAT.NS",
    "DEEPAKNTR.NS",
    "ESCORTS.NS",
    "EXIDEIND.NS",
    "FEDERALBNK.NS",
    "FORTIS.NS",
    "GLENMARK.NS",
    "GUJGASLTD.NS",
    "HINDPETRO.NS",
    "IDBI.NS",
    "INDUSTOWER.NS",
    "JUBLFOOD.NS",
    "KEI.NS",
    "KAYNES.NS",
    "LICHSGFIN.NS",
    "MANAPPURAM.NS",
    "MFSL.NS",
    "MANKIND.NS",
    "MINDTREE.NS",
    "NATIONALUM.NS",
    "NMDC.NS",
    "OIL.NS",
    "PAGEIND.NS",
    "PEL.NS",
    "PETRONET.NS",
    "PIIND.NS",
    "PRESTIGE.NS",
    "SAIL.NS",
    "SCHAEFFLER.NS",
    "SOLARINDS.NS",
    "SUPREMEIND.NS",
    "TATAELXSI.NS",
    "THERMAX.NS",
    "TORNTPOWER.NS",
    "UNOMINDA.NS",
    "VBL.NS",
    "VOLTAS.NS"
]


UNIVERSES = {
    "NIFTY_50": NIFTY_50,
    "NIFTY_100": NIFTY_50 + NIFTY_NEXT_50,
    "NIFTY_150": NIFTY_50 + NIFTY_NEXT_50 + EXTENDED_50
}


class UniverseEngine:
    def __init__(self):
        self.universes = {
            name: list(dict.fromkeys(symbols))
            for name, symbols in UNIVERSES.items()
        }

    def get_universe(self, universe_name):
        key = universe_name.upper()

        if key not in self.universes:
            raise ValueError(
                f"Unknown universe '{universe_name}'. "
                f"Available universes: {list(self.universes.keys())}"
            )

        return self.universes[key]

    def get_symbols(self, universe_name):
        return self.get_universe(universe_name)

    def get_count(self, universe_name):
        return len(self.get_universe(universe_name))

    def get_available_universes(self):
        return {
            name: len(symbols)
            for name, symbols in self.universes.items()
        }

    def validate(self, universe_name):
        symbols = self.get_universe(universe_name)

        if not symbols:
            raise ValueError(f"{universe_name} universe is empty")

        if len(symbols) != len(set(symbols)):
            raise ValueError(f"{universe_name} contains duplicate symbols")

        invalid = [
            symbol for symbol in symbols
            if not symbol.endswith(".NS")
        ]

        if invalid:
            raise ValueError(
                f"Invalid NSE symbols detected: {invalid}"
            )

        return True

    def summary(self):
        print("=" * 70)
        print("QUANTFORGE UNIVERSE ENGINE")
        print("=" * 70)

        for name, symbols in self.universes.items():
            print(f"{name:<15} {len(symbols):>4} stocks")

        print("=" * 70)


if __name__ == "__main__":
    engine = UniverseEngine()
    engine.summary()

    for universe in engine.universes:
        engine.validate(universe)

    print("UNIVERSE VALIDATION: PASSED")