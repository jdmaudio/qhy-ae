# qhy-ae

Python script which captures video from QHY cam using pysky360.QHYCamera and automatically adjusts exposure and gain settings to achieve a desired mean sample value (MSV) for the captured frames. The algorithm uses proportional-integral (PI) control based on the image histogram. 

When decreasing MSV, gain is prioritized to be reduced before adjusting exposure.
