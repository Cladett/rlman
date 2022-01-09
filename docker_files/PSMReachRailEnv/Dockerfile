FROM nvidia/cuda:10.0-base

# Install packages
RUN apt update 
RUN apt install -y --no-install-recommends vim tar xz-utils libx11-6 libxcb1 libxau6 libgl1-mesa-dev xvfb dbus-x11 x11-utils libxkbcommon-x11-0 libavcodec-dev libavformat-dev libswscale-dev wget libzmq3-dev
RUN apt autoremove -y
RUN apt autoclean -y

# Download CoppeliaSim
RUN mkdir -p /shared /opt
RUN wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz -P /opt
RUN tar xf /opt/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz -C /opt 
RUN rm /opt/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04.tar.xz

# Setup environment variables for Coppelia
ENV COPPELIASIM_ROOT_DIR=/opt/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
ENV LD_LIBRARY_PATH=$COPPELIASIM_ROOT_DIR:$LD_LIBRARY_PATH
ENV PATH=$COPPELIASIM_ROOT_DIR:$PATH

# Expose the ports used in the scene
EXPOSE 80
EXPOSE 19996
EXPOSE 19997
EXPOSE 19998
EXPOSE 19999

#Copy in our scene
# CLAUDIA adding enviroment with different object
#ADD dVRK-oneArm-pick_KID.ttt /opt/scene.ttt
ADD dVRK-oneArm-pick_rail.ttt /opt/scene.ttt
# Testing the responsive rail 
# ADD dVRK-oneArm-pick_respon_rail.ttt /opt/scene.ttt


# Run CoppeliaSim
CMD $COPPELIASIM_ROOT_DIR/coppeliaSim -s -h -q /opt/scene.ttt
