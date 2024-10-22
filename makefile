docker-build:
	docker-compose -f docker-compose.yml up --build -d --remove-orphans

docker-up:
	docker-compose -f docker-compose.yml up -d --remove-orphans

docker-start: | docker-up

docker-down:
	docker-compose -f docker-compose.yml down --remove-orphans

docker-stop:
	docker-compose -f docker-compose.yml stop

login:
	docker-compose -f docker-compose.yml exec ml-tutorial bash
