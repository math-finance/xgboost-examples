from twisted.web.server import Site
from twisted.web.static import File
from twisted.internet import reactor

def run(path):
	reactor.listenTCP(2821, Site(File(path=path, defaultType='application/octet-stream')))
	reactor.run()