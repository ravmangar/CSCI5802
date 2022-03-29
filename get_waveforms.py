#!/bin/env/python3

import obspy
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader


class MINISEED_Downloader:
    def __init__(self, min_lat=30, max_lat=50, min_long=5, max_long=35, start_time="01-01-2020", end_time="12-01-2020"):
        self.min_lat = min_lat
        self.min_long = min_long
        self.max_long = max_long
        self.max_lat = max_lat
        self.start_time = self.UTCTime(start_time)
        self.end_time = self.UTCTime(end_time)

        self.restrictions = Restrictions(
            starttime=self.start_time,
            endtime=self.end_time,
            reject_channels_with_gaps=True,
            minimum_length=0.9,
            minimum_interstation_distance_in_m=1000,
            channel_priorities=["HH[ZNE]", "BH[ZNE]"],
            location_priorities=["", "00", "01"]
        )

        self.area = RectangularDomain(
            minlatitude=self.min_lat,
            maxlatitude=self.max_lat,
            minlongitude=self.min_long,
            maxlongitude=self.max_long
        )

        self.mdl = MassDownloader()

    def UTCTime(self, _time):
        time = _time.split("-")
        day = int(time[0])
        month = int(time[1])
        year = int(time[2])

        return obspy.UTCDateTime(year=year, month=month, day=day)

    def download(self):
        self.mdl.download(self.area, self.restrictions, mseed_storage="waveforms", stationxml_storage="stations",
                          threads_per_client=6)