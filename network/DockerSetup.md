<h1>SIWN - Network Docker Container</h1>

<p  style='line-height:115%'><span style='mso-ascii-font-family:
Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:Aptos;mso-bidi-font-family:
Aptos'><o:p>&nbsp;</o:p></span></p>

<p  style='line-height:115%'><span style='mso-ascii-font-family:
Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:Aptos;mso-bidi-font-family:
Aptos'>Secure Infrastructure-less Wireless Networks (SIWNs) is a technology
that allows a set of clients to exchange information securely through an ad-hoc
wireless network. SIWNs provide security through client authentication, zero
knowledge key generation for data encryption, and ensuring network trust
through blockchain enabled network mapping.<o:p></o:p></span></p>

<p >This document includes the structure and requirements for
the development of a docker container that contains the necessary requirements
for the deployment of the network component of the SIWN</p>

<p ><o:p>&nbsp;</o:p></p>

<h2>References</h2>

<p ><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><a
href="https://hub.docker.com/repository/docker/joseasanchezviloria/siwn-radio/general">SIWN
radio docker container</a></p>

<p ><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><a
href="https://docs.docker.com/guides/walkthroughs/what-is-a-container/">Docker
containers</a></p>

<p ><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><a
href="https://aws.amazon.com/what-is/osi-model/">OSI model – network layers</a></p>

<p ><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><a href="https://www.gnuradio.org">GNU-radio</a></p>

<p ><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><a
href="https://github.com/C2A2-at-Florida-Atlantic-University/siwn-node/tree/main/network">SIWN
network repo</a></p>

<p ><o:p>&nbsp;</o:p></p>

<h2>Background</h2>

<p class=MsoListParagraphCxSpFirst style='text-indent:-.25in;mso-list:l0 level1 lfo2'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos;color:black;mso-themecolor:text1'><span
style='mso-list:Ignore'>-<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>Docker: <span style='color:black;mso-themecolor:
text1'>Docker is an open-source platform for developing, deploying, and running
applications. It uses a concept called containerization to package up an
application with all its dependencies into standardized units called
containers.<o:p></o:p></span></p>

<p class=MsoListParagraphCxSpLast style='text-indent:-.25in;mso-list:l0 level1 lfo2'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos;color:black;mso-themecolor:text1'><span
style='mso-list:Ignore'>-<span style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]><span style='color:black;mso-themecolor:text1'>Container:
Containers have everything that your code needs <span>in order to</span>
run, down to a base operating system.<o:p></o:p></span></p>

<p ><o:p>&nbsp;</o:p></p>

<h2>Use cases</h2>

<p >The SIWN network docker container will be able to run the
software that oversees the network operations of a SIWN node. This includes all
software to deploy a complete network solution as described in the OSI model.
This includes physical, data, network, transport, session, presentation, and
application layers.</p>

<p ><o:p>&nbsp;</o:p></p>

<h2>Requirements</h2>

<p class=MsoListParagraphCxSpFirst style='text-indent:-.25in;mso-list:l3 level1 lfo1'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>This should run in an embedded system</p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in;mso-list:l3 level1 lfo1'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>Ubuntu 20</p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in;mso-list:l3 level1 lfo1'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>They should use off the shelf SDR</p>

<p class=MsoListParagraphCxSpMiddle style='text-indent:-.25in;mso-list:l3 level1 lfo1'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>Using common radio development software such as
GNU radio</p>

<p class=MsoListParagraphCxSpLast style='text-indent:-.25in;mso-list:l3 level1 lfo1'><![if !supportLists]><span
style='mso-ascii-font-family:Aptos;mso-fareast-font-family:Aptos;mso-hansi-font-family:
Aptos;mso-bidi-font-family:Aptos'><span style='mso-list:Ignore'>-<span
style='font:7.0pt "Times New Roman"'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</span></span></span><![endif]>Use an API that enables the communication with
other components</p>

<p ><o:p>&nbsp;</o:p></p>

<h2>Content</h2>

<p >To perform the network operations while also enabling the
capability to customize them, we will be developing our own custom network
through SDRs and open-source software for managing these resources. GNU Radio
is a free &amp; open-source software development toolkit that provides signal
processing blocks to implement software radios. You can use it to write
applications to receive data out of digital streams or to push data into
digital streams, which is then transmitted using hardware. This software also
includes the capability to simulate such use cases. GNU radio also enables us
to quickly develop software by providing the source code templates to perform
the radio operations. We will be using these templates to customize them and
write our own code for our required physical layer operations. The provided
templates are available in C++ and python. We will be using C++ as the core
language for developing our network solutions.</p>

<p >As of today, these operations have been written in python as
it was the language the team was most comfortable in the SIWN node repo. The
GNU version selected to perform the PHY network operations is 3.8.5 which is
compliant with the OS version which is Ubuntu 20.04 and the python version 3.6.</p>

<p >The docker container will be available in Jose Sanchez <a
href="https://hub.docker.com/repository/docker/joseasanchezviloria/siwn-radio/general">docker
repo</a> and should at some point be under some common repo for this project.
The following are some operations for pulling, running, starting, and executing
the docker container.</p>

<p ><o:p>&nbsp;</o:p></p>

<p >Pull Docker Container </p>

<p >docker image pull <span class=SpellE>joseasanchezviloria</span>/siwn-<span
class=GramE>radio:v</span>1 </p>

<p ><span style='mso-spacerun:yes'> </span></p>

<p >Run docker container: </p>

<p >docker run --name <span class=SpellE>siwn</span>-network --runtime
<span class=SpellE>nvidia</span> -it -e DISPLAY=$DISPLAY -v ~/<span
class=SpellE>siwn</span>:/home/<span class=SpellE>siwn</span> -v
/dev/net/tun:/dev/net/tun --network host --cap-add=NET_ADMIN <span
class=SpellE>joseasanchezviloria</span>/siwn-<span>radio:v</span>1 </p>

<p ><span style='mso-spacerun:yes'> </span></p>

<p >Start Docker container: </p>

<p >docker start <span class=SpellE>siwn</span>-network</p>

<p ><span style='mso-spacerun:yes'> </span></p>

<p >Stop docker container: </p>

<p >docker stop <span class=SpellE>siwn</span>-network</p>

<p ><span style='mso-spacerun:yes'> </span></p>

<p >Execute (access) Docker container </p>

<p >docker exec –it <span class=SpellE>siwn</span>-network
/bin/bash</p>

<p ><o:p>&nbsp;</o:p></p>

<h2>Other approaches considered</h2>

<p >It was also considered to create environments instead of containers,
but containers <span>is</span> a more suitable approach since we
needed to containerize OS specific dependencies other than just python or
language specific dependencies.</p>
